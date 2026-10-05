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
//!   the head's own copy of `M`'s gains `γ`, which are part of the law and not trained. Query heads
//!   that share a key and value in `M` (grouped-query attention) read one shared `K` and `V` here
//!   too, so the explanation stays in the family `M` realizes.
//! * An MLP's block reads its layer's second normed stream and writes the MLP's output. Its
//!   functions are `f_i(x) = φ(g_i·x + c_i) u_i`, with the native pointwise law `φ`,
//!   a gate direction `g_i`, a gate bias `c_i` and an output `u_i`; for a gated MLP (SwiGLU on
//!   Qwen3) `f_i(x) = φ(g_i·x) (b_i·x) u_i` with an up direction `b_i`. Biases appear only where
//!   `M` has them, so a bias-free MLP maps 0 to 0.
//!
//! The library starts at `M`: native neuron `i` is function `i` with its original
//! activation and coefficients, and a head is its own function.
//!
//! # The code length
//!
//! The library's parameters `θ` are partitioned into prior groups `G`: a key-value group's rotary
//! plane (the plane's rows of the shared key and of every query head's query), its value
//! coordinate (one row of the shared value), an MLP function's gate `(g_i, c_i)`, its up direction `b_i` when gated, and its output `u_i`. With the posterior `q(θ) = Π_j N(μ_j, σ_j²)` and the
//! prior `p(θ_G) = N(0, v_G I)`, the code length in nats is
//!
//! `F = E_q[D(θ)] + Σ_G KL(q_G ‖ p_G) + Σ_{active G} (½ ln |G| + L_scale(v_G)) + L_subset`,
//!
//! the bits-back code of the training data given the explanation plus the explanation. `D(θ)`
//! sums `KL(M_e ‖ P_e)` of the next-token distributions over every token of every training
//! experiment `e` (below), so the data term's weight is the number of scored tokens `N` and no
//! tradeoff weight exists. `N` is the amount of `M`'s behaviour the explanation is asked to
//! explain; it is the one choice left, and results are reported along it. Each prior variance is chosen by empirical Bayes in closed form,
//! `v_G = (1/|G|) Σ_{j∈G} (μ_j² + σ_j²)`, which makes `KL(q_G ‖ p_G) = ½ (|G| ln v_G − Σ_{j∈G}
//! ln σ_j²)` with derivatives `μ_j / v_G` in `μ_j` and `σ_j² / v_G − 1` in `ln σ_j`. An active
//! group sends its variance at the precision of a parameter estimated from `|G|` values,
//! `½ log2 |G|` bits (the two-part code's parameter precision), after its scale: the integer
//! exponent `round(log2(v_G / v⁰_G))` relative to the variance `v⁰_G` of the group's native
//! starting values, which every decoder has from `M`, in the Elias δ code of its signed index
//! (`L_scale`; the precision prices the fraction, the scale the exponent, since `(μ, σ) → (a μ, a
//! σ)` leaves `KL` unchanged). Which groups are in the explanation is sent once in the enumerative
//! subset code, `L_subset = L_int(k + 1) + ⌈log2 C(n, k)⌉` bits for `k` of `n` groups. A group the data does not inform
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
//! The data are interchange experiments (`interchange`), one fixed collection drawn once from the
//! seed: per training base sequence, one clean experiment and one patched experiment under the
//! same hybrid, with the stated weights of `interchange::sample` (`P` alone or a random
//! block-subset hybrid with probability ½ each; the patched block uniform over the `2L` blocks,
//! then with probability ½ one read variable uniform within the block, else a joint read patch of
//! a random subset of the block's variables, each included with probability ½; then a position).
//! `M`'s reads span every block's whole stream, so the complement of `M`'s reads is empty; the
//! collection covers every one of `M`'s reads instead, those `P` removes included, one at a time
//! and jointly. A base's source is another training sequence, drawn uniformly. The variables
//! and patch directions are `M`'s own reads (the library's start): the questions do not move as
//! `P` learns or loses functions, so `F`, the convergence test and every removal comparison score
//! the same experiments, and `M`'s targets for them are made once per batch and kept
//! (`interchange::Targets`, one shard per batch next to the checkpoint). Experiments whose
//! directions are `P`'s current reads are a separate held-out report, never the objective.
//!
//! # The fit
//!
//! Each step draws one weight sample `θ = μ + σ ⊙ ε` (`ε` standard normal), runs a batch's
//! experiments, and takes one step of the improved variational online Newton method (IVON; Shen et
//! al., ICML 2024) on `F / N = E_q[ℓ] + KL(q ‖ p) / N`, `ℓ` the data term per scored token and `N`
//! the scored tokens of every training experiment: the batch's gradient `g` of `ℓ` at the sample
//! gives the curvature estimate `ĥ = g ε / σ`, whose average over the last epoch's batches is `h`
//! (`β₂ = 1 − 1/B` for `B` training batches); the posterior's standard deviation is then
//! `σ = 1 / √(N (h + δ))` with `δ = 1 / (N v_G)` the group prior's precision per token, the value
//! at which `F` is stationary in `σ`, so `σ` has no step size; the mean takes the preconditioned
//! step `α (m + δ μ) / (h + δ)` along the gradient's momentum `m`. The posterior stays on the
//! device through an epoch (`device_posterior`): the sample is written into the explanation's
//! program, the gradient stays where the reverse pass left it, and the IVON step and the groups'
//! divergences run there; the host holds it between epochs, for the
//! held-out evaluation, the checkpoint and the removal step. An epoch visits every training batch
//! once, in a fixed order. The continuous fit
//! has converged when an epoch's mean improvement of the per-batch objective estimate over the
//! previous epoch, paired by batch, is smaller than its standard error.
//!
//! A converged fit then proposes removals, each accepted only when it does not increase `F`,
//! estimated over the whole training set with one common weight sample per batch for both sides of
//! every comparison, on the fixed collection ([`Removal`]):
//! * the groups with no path to the output (the planes of a head none of whose value coordinates
//!   is active; the other groups of an MLP function one of whose groups is removed, since every
//!   pointwise law is zero at zero), all at once;
//! * then prefixes of the remaining groups in increasing order of each one's second-order removal
//!   effect `½ Σ_j (μ_j² (1/σ_j² − 1/v_G) − 1 + σ_j²/v_G) − KL_G − ½ ln |G|` (zeroing the mean and
//!   the noise at the curvature `1/σ² − 1/v_G` that IVON's `σ` holds, less the description it
//!   saves), searched by bisection, `O(log n)` full training evaluations for `n` groups;
//! * when no prefix is accepted, single groups: the [`SINGLES`] of lowest estimated effect and
//!   [`SINGLES`] drawn at random, the best accepted.
//! Where a proposal deletes functions of an MLP, the MLP's surviving functions' outputs move by the
//! least-squares solution that takes over the deleted functions' output on `P`'s own states
//! (`library_compensation`), and the comparison scores the proposal with those outputs. The order
//! and the search are proposals; acceptance never increases the sampled objective. Removal effects
//! can cancel, so the objective need not be monotone in the prefix length. The fit alternates
//! converging and removing; it stops when a round accepts nothing, which says the search found no
//! removal, not that none exists ([`Outcome::Exhausted`]).
//!
//! # Evaluation
//!
//! At the start and after every epoch a fixed subset of the held-out sequences (the first batch of
//! them) is scored ([`HeldOut`]), and all of them only while the evaluations so far, with the last
//! full evaluation's time (estimated from the subset's until one is made), stay within
//! [`EVALUATION_SHARE`] of the training time so far; the fit ends with a full evaluation. The
//! held-out experiments are a fixed sample from the training distribution (`interchange::sample`,
//! from each held-out batch's seed): per base one clean and one patched experiment, sources among
//! the held-out sequences. `F` per token is the
//! held-out data term at one weight sample per batch plus the description spread over the training
//! experiments' scored tokens; the divergences per experiment kind are taken at the posterior mean.
//! Per layer it counts the surviving heads and MLP functions and, per token, the functions whose
//! activation is not zero and those whose activation exceeds its posterior noise scale.
//!
//! The explanation is reported at the posterior mean `μ` ([`posterior_mean`]); removed groups are
//! absent only when they fill complete interface blocks.

use crate::{
    artifact::{Argument, Artifact, Callee, Owner},
    device_posterior::{DevicePosterior, Ivon},
    interchange::{self, Batch, Experiment, Interchange, Patch, ReadVariable, Targets},
    library_compensation::Compensation,
    operator_program::{
        FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram, Provenance, Rule, SequenceLayout, SlotValues,
        exact_precision,
    },
    run_check::{LayerNodes, head_projection},
};
use gam_gpu::tensor::{Device, Op, Tensor};
use gam_runtime::warm_start::Fingerprinter;
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    f64::consts::LN_2,
    io::{BufReader, Read, Write},
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

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
    /// Groups that start outside the explanation, exactly zero (the output a read–write tie
    /// replaces, `library_sharing`).
    pub removed: Vec<usize>,
    /// The nats of the explanation's discrete choices: each exact read–write tie's choice of the
    /// write it reads among its candidates (`library_mixture`).
    pub fixed_nats: f64,
}

/// A prior term beyond the groups' own Gaussian priors (`library_mixture`): its value at a weight
/// sample enters `F` beside the groups' divergences, and its gradient in the sample joins the data
/// term's.
pub trait PriorTerm {
    /// The trainable operators it reads (indices into `Explanation::trainable`).
    fn operators(&self) -> Vec<usize>;
    /// Re-choose its structure from `posterior`, once per epoch.
    fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String>;
    /// Its value in nats at the weight sample `theta` (its operators', by trainable index) of
    /// `posterior`, its gradient in `theta`, and with `learn` one step of its own parameters.
    fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String>;
    /// The nats of the parameters it sends.
    fn cost(&self, posterior: &Posterior) -> Result<f64, String>;
    /// Its state, for a checkpoint, and its state restored from one.
    fn save(&self) -> Result<serde_json::Value, String>;
    fn load(&mut self, value: &serde_json::Value) -> Result<(), String>;
}

/// The weight sample of `key` (`DevicePosterior::sample_into`'s draws) of `posterior`'s trainable
/// operators `operators`, on the host.
fn host_sample(posterior: &Posterior, operators: &[usize], key: u64) -> BTreeMap<usize, Array2<f64>> {
    operators
        .par_iter()
        .map(|&i| {
            let (mean, log_sd) = (&posterior.mean[i], &posterior.log_sd[i]);
            let cols = mean.ncols();
            let theta = Array2::from_shape_fn(mean.dim(), |(r, c)| {
                let s = log_sd[[r, c]];
                if s == f64::NEG_INFINITY { mean[[r, c]] } else { mean[[r, c]] + s.exp() * f64::from(gam_gpu::tensor::posterior_normal(key, i as u64, (r * cols + c) as u64)) }
            });
            (i, theta)
        })
        .collect()
}

/// `prior`'s value with the parameters it sends, in nats, at the weight sample of `key` of
/// `posterior`, and its gradient in the sample; with `learn`, one step of its own parameters.
fn prior_term(prior: &mut (dyn PriorTerm + 'static), posterior: &Posterior, key: u64, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
    let theta = host_sample(posterior, &prior.operators(), key);
    let (value, gradient) = prior.sample(posterior, &theta, learn)?;
    Ok((value + prior.cost(posterior)?, gradient))
}

/// One layer of the explanation: its native sites (`run_check::layer_nodes`) and the prior groups
/// of its functions.
#[derive(Clone, Debug)]
pub struct Layer {
    pub sites: LayerNodes,
    /// Per head, its planes' groups and its value coordinates' groups (those of its key-value
    /// group, shared with the group's other query heads).
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
fn affine_map(native: &OperatorProgram, node: usize, input: usize) -> Result<(Arc<Operator>, Option<Arc<Operator>>), String> {
    match &native.nodes[node] {
        Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == input => {
            Ok((Arc::clone(&native.operators[terms[0].1]), bias.map(|b| Arc::clone(&native.operators[b]))))
        }
        other => Err(format!("node {node} is not an affine map of node {input}: {other:?}")),
    }
}

/// The library block of a gated MLP `down(φ(A x + c) ⊙ (B x + e))` (SwiGLU on Qwen3): functions
/// `f_i(x) = φ(a_i·x + c_i) (b_i·x + e_i) u_i` with `M`'s law `φ`, its gate `a_i`, up direction
/// `b_i` and output `u_i`, and the biases `c_i`, `e_i` only where `M` has them.
fn gated_mlp(native: &OperatorProgram, artifact: Artifact, layer: &LayerNodes, l: usize, left: usize, right: usize) -> Result<(Artifact, Vec<Owner>), String> {
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
    let owners = mlp_owners(&name, units.width(), &gate, gate_bias.as_deref(), Some((&up, up_bias.as_deref())), &down);
    let mut bias = |values: Option<Arc<Operator>>, part: &str, source: &str| -> Result<Option<usize>, String> {
        let Some(values) = values else { return Ok(None) };
        operators.push(library_operator(&format!("{name}.{part}_bias"), units.clone(), Interface::constant(), values.matrix(), source)?);
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
    Ok((artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(x)], layer.mlp, operators)?, owners))
}

/// The owners of an MLP block `name` of `units` functions (`artifact::Owner`): per function its
/// gate row (and bias), its up row (and bias) when gated, and its output column, each replacing
/// the same row or column of `M`'s map.
fn mlp_owners(name: &str, units: usize, gate: &Operator, gate_bias: Option<&Operator>, up: Option<(&Operator, Option<&Operator>)>, down: &Operator) -> Vec<Owner> {
    let mut out = Vec::new();
    let mut add = |part: &str, native: &Operator, unit: usize, column: bool| {
        let (rows, cols) = if column { (0..native.rows.width(), unit..unit + 1) } else { (unit..unit + 1, 0..native.cols.width()) };
        out.push(Owner {
            operator: format!("{name}.{part}"),
            rows: rows.clone(),
            cols: cols.clone(),
            body: name.to_string(),
            site: name.to_string(),
            native: native.name.clone(),
            native_rows: rows,
            native_cols: cols,
            ..Owner::default()
        });
    };
    for i in 0..units {
        add("gate", gate, i, false);
        if let Some(bias) = gate_bias {
            add("gate_bias", bias, i, false);
        }
        if let Some((up, up_bias)) = up {
            add("up", up, i, false);
            if let Some(bias) = up_bias {
                add("up_bias", bias, i, false);
            }
        }
        add("out", down, i, true);
    }
    out
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
    let mut owners = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        // Per key-value group (the native key and value projections its query heads read): its
        // name, its query heads, its planes and its value width.
        let mut groups: Vec<(usize, usize, String, Vec<(usize, String)>, Vec<Vec<usize>>, usize)> = Vec::new();
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
            let mut operators = vec![library_operator(&format!("{name}.q"), coordinates.clone(), q.cols.clone(), q.matrix(), &q.name)?];
            // Query heads that read one key and value in `M` read one key map and one value map:
            // one posterior, every head's use in its gradient, one divergence.
            let group = groups.iter().position(|g| g.0 == key && g.1 == value);
            let (k_op, v_op) = match group {
                Some(g) => (index_of(&artifact.program, &format!("{}.k", groups[g].2))?, index_of(&artifact.program, &format!("{}.v", groups[g].2))?),
                None => {
                    let shared = format!("library.l{l}.kv{}", groups.len());
                    operators.push(library_operator(&format!("{shared}.k"), coordinates.clone(), k.cols.clone(), k.matrix(), &k.name)?);
                    operators.push(library_operator(&format!("{shared}.v"), v.rows.clone(), v.cols.clone(), v.matrix(), &v.name)?);
                    (base + 1, base + 2)
                }
            };
            let mut nodes = vec![
                Node::Param { index: 0 },
                Node::Affine { terms: vec![(0, base)], bias: None },
                Node::Affine { terms: vec![(0, k_op)], bias: None },
                Node::Affine { terms: vec![(0, v_op)], bias: None },
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
            // The head's query map, and its group's key and value maps as this head reads them.
            let shared = match group {
                Some(g) => groups[g].2.clone(),
                None => format!("library.l{l}.kv{}", groups.len()),
            };
            for (operator, source) in [(format!("{name}.q"), &q), (format!("{shared}.k"), &k), (format!("{shared}.v"), &v)] {
                let (rows, cols) = (0..source.rows.width(), 0..source.cols.width());
                owners.push(Owner { operator, rows: rows.clone(), cols: cols.clone(), body: name.clone(), site: name.clone(), native: source.name.clone(), native_rows: rows, native_cols: cols, ..Owner::default() });
            }
            let pairs: Vec<Vec<usize>> = match rotary {
                Some(r) => {
                    let pairs = r.pairs();
                    let rotated: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                    pairs.iter().map(|&(a, b)| vec![a, b]).chain((0..q.rows.width()).filter(|c| !rotated.contains(c)).map(|c| vec![c])).collect()
                }
                None => (0..q.rows.width()).map(|c| vec![c]).collect(),
            };
            match group {
                Some(g) if groups[g].4 == pairs && groups[g].5 == v.rows.width() => groups[g].3.push((h, name)),
                Some(_) => return Err(format!("layer {l} head {h}: the query heads of one key-value group differ in their planes")),
                None => groups.push((key, value, format!("library.l{l}.kv{}", groups.len()), vec![(h, name)], pairs, v.rows.width())),
            }
        }
        planes.extend(groups.into_iter().map(|(_, _, shared, heads, pairs, values)| (l, shared, heads, pairs, values)));
        if let Node::Hadamard { left, right } = native.nodes[layer.active] {
            let (built, more) = gated_mlp(native, artifact, layer, l, left, right)?;
            artifact = built;
            owners.extend(more);
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
        let name = format!("library.l{l}.mlp");
        owners.extend(mlp_owners(&name, units.width(), &up, up_bias.map(|b| native.operators[b].as_ref()), None, &down));
        let base = artifact.program.operators.len();
        let mut operators = vec![
            library_operator(&format!("{name}.gate"), units.clone(), up.cols.clone(), up.matrix(), &up.name)?,
            library_operator(&format!("{name}.out"), down.rows.clone(), units.clone(), down.matrix(), &down.name)?,
        ];
        // A gate bias only where `M` has one, so a bias-free MLP still maps 0 to 0.
        let gate_bias = match up_bias {
            Some(op) => {
                operators.push(library_operator(&format!("{name}.gate_bias"), units.clone(), Interface::constant(), native.operators[op].matrix(), &up.name)?);
                Some(base + 2)
            }
            None => None,
        };
        let rule = Rule {
            name: name.clone(),
            inputs: vec![up.cols.clone()],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Affine { terms: vec![(0, base)], bias: gate_bias },
                Node::Pointwise { input: 1, laws: laws.clone() },
                Node::Affine { terms: vec![(2, base + 1)], bias: None },
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
    // Per key-value group, a group per rotary plane holding the plane's rows of the shared key
    // and of every query head's query, and a group per value coordinate of the shared value.
    let mut heads: Vec<Vec<Option<(Vec<usize>, Vec<usize>)>>> = layers.iter().map(|l| vec![None; l.reads.len()]).collect();
    for (l, shared, members, pairs, values) in &planes {
        let (k, v) = (index_of(program, &format!("{shared}.k"))?, index_of(program, &format!("{shared}.v"))?);
        let queries = members.iter().map(|(_, name)| index_of(program, &format!("{name}.q"))).collect::<Result<Vec<_>, _>>()?;
        let d = program.operators[k].cols.width();
        let first = groups.len();
        for (p, rows) in pairs.iter().enumerate() {
            let cells = queries.iter().chain([&k]).map(|&operator| Cells { operator, rows: rows.clone(), cols: 0..d }).collect();
            groups.push(Group { name: format!("{shared}.plane{p}"), cells });
        }
        for j in 0..*values {
            groups.push(Group { name: format!("{shared}.value{j}"), cells: vec![Cells { operator: v, rows: vec![j], cols: 0..d }] });
        }
        for (h, _) in members {
            heads[*l][*h] = Some(((first..first + pairs.len()).collect(), (first + pairs.len()..groups.len()).collect()));
        }
        trainable.extend(queries.into_iter().chain([k, v]));
    }
    for (layer, heads) in out.iter_mut().zip(heads) {
        layer.heads = heads.into_iter().collect::<Option<Vec<_>>>().ok_or("a head in no key-value group")?;
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
    artifact.owners = owners;
    Ok(Explanation { artifact, trainable, groups, layers: out, removed: Vec::new(), fixed_nats: 0.0 })
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
    /// Per group, the variance `v⁰_G` of its native starting values, against which its variance's
    /// scale is sent (zero for a group that starts outside the explanation).
    initial: Vec<f64>,
}

/// The bits of a group's variance scale (module note): the integer exponent
/// `round(log2(v_G / v⁰_G))` in the Elias δ code of its signed index.
fn scale_bits(variance: f64, initial: f64) -> f64 {
    let exponent = (variance / initial).log2().round();
    if !exponent.is_finite() {
        return f64::INFINITY;
    }
    // A finite ratio of finite positive variances has |exponent| ≤ 2098, well inside i64.
    match crate::codec::signed_codeword_argument(exponent as i64).and_then(crate::codec::elias_delta_len_bits) {
        Ok(bits) => bits as f64,
        Err(_) => f64::INFINITY,
    }
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
        if explanation.removed.iter().any(|g| *g >= explanation.groups.len()) {
            return Err("a removed group the explanation does not have".into());
        }
        if let Some(g) = (0..squares.len()).find(|g| !explanation.removed.contains(g) && !(squares[*g].1 > 0.0 && squares[*g].1.is_finite())) {
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
        let initial = squares.iter().map(|(count, sum)| if *count > 0.0 { sum / count } else { 0.0 }).collect();
        let mut posterior = Self { mean, log_sd, active: vec![true; explanation.groups.len()], membership, spans, initial };
        posterior.remove(&explanation.removed);
        Ok(posterior)
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

    /// Per group, the second-order estimate of `F`'s change in nats when it alone is removed
    /// (module note): the data term's rise from zeroing its means and noise at the curvature
    /// `1/σ² − 1/v_G`, less the description it saves (zero for a removed group).
    fn removal_estimates(&self) -> Vec<f64> {
        let moments = self.moments();
        let costs = self.costs();
        let partial: Vec<(Range<usize>, Vec<f64>)> = (0..self.mean.len())
            .into_par_iter()
            .map(|i| {
                let span = self.spans[i].clone();
                let mut local = vec![0.0; span.len()];
                for ((mu, s), group) in self.mean[i].iter().zip(self.log_sd[i].iter()).zip(self.membership[i].iter()) {
                    let g = *group as usize;
                    if self.active[g] {
                        let (variance, sd2) = (moments[g].second / moments[g].count, (2.0 * s).exp());
                        local[g - span.start] += 0.5 * (mu * mu * (1.0 / sd2 - 1.0 / variance) - 1.0 + sd2 / variance);
                    }
                }
                (span, local)
            })
            .collect();
        let mut out = vec![0.0; self.active.len()];
        for (span, local) in partial {
            for (g, e) in span.zip(local) {
                out[g] += e;
            }
        }
        out.iter_mut().zip(&costs).zip(&self.active).for_each(|((e, c), active)| *e = if *active { *e - c } else { 0.0 });
        out
    }

    /// Per group, `KL(q_G ‖ p_G)` in nats (zero for a removed group).
    pub fn divergences(&self) -> Vec<f64> {
        self.moments().iter().zip(&self.active).map(|(m, active)| if *active { m.divergence() } else { 0.0 }).collect()
    }

    /// Per group, `KL(q_G ‖ p_G)` plus its variance's `½ ln |G|`, in nats (zero for a removed
    /// group).
    pub fn costs(&self) -> Vec<f64> {
        self.moments()
            .iter()
            .zip(&self.active)
            .zip(&self.initial)
            .map(|((m, active), initial)| if *active { m.divergence() + 0.5 * m.count.ln() + scale_bits(m.second / m.count, *initial) * LN_2 } else { 0.0 })
            .collect()
    }

    /// The nats of which groups are in the explanation (module note).
    pub fn subset_nats(&self) -> f64 {
        let active = self.active.iter().filter(|a| **a).count();
        crate::codec::subset_code_len_bits(self.active.len(), active).map_or(f64::INFINITY, |bits| bits as f64 * LN_2)
    }

    /// `Σ_G KL(q_G ‖ p_G)`, the active groups' variances and which groups are active, in nats.
    pub fn description(&self) -> f64 {
        self.costs().iter().sum::<f64>() + self.subset_nats()
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
    pub fn remove(&mut self, groups: &[usize]) {
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
    /// IVON's step in `μ` as a fraction of the Newton step `(m + δ μ) / (h + δ)`, and the decay of
    /// the gradient's momentum `m`. The curvature estimate averages over one epoch's batches and
    /// sets `σ` (module note).
    pub rate: f64,
    pub beta1: f64,
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
            || !positive(self.rate)
            || !(0.0..1.0).contains(&self.beta1)
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
    /// `Σ_G KL(q_G ‖ p_G)`, and the active groups' variances (precision and scale) with the code of
    /// which groups are active, in bits.
    pub divergence_bits: f64,
    pub variance_bits: f64,
    /// The explanation's discrete choices (`Explanation::fixed_nats`), and the prior term's value at
    /// one weight sample with the parameters it sends (`PriorTerm`), in bits.
    #[serde(default)]
    pub choice_bits: f64,
    #[serde(default)]
    pub prior_bits: f64,
    /// At the posterior mean: the sample's clean and patched experiments per hybrid size `k = 1..=2L` (`k = 2L`
    /// is the explanation alone), and the patched ones by kind: one read variable, or a joint
    /// patch of several.
    pub clean: Vec<Option<f64>>,
    pub patched: Vec<Option<f64>>,
    pub read_patch: Option<f64>,
    pub joint_patch: Option<f64>,
    /// The patched experiments with directions at `P`'s own reads (the posterior mean) instead of
    /// `M`'s: adaptive questions, reported apart and never trained on.
    pub adaptive_patch: Option<f64>,
    pub layers: Vec<LayerCount>,
}

/// The largest share of the fit's training time its per-epoch held-out evaluations may take: the
/// full held-out set is scored after an epoch only while the evaluations stay within it.
pub const EVALUATION_SHARE: f64 = 0.1;

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
    /// The epoch's training seconds.
    pub seconds: f64,
    /// The held-out evaluation after the epoch's steps on the fixed subset, and on every held-out
    /// sequence when the schedule ran it (module note).
    pub held_out: HeldOut,
    pub held_out_full: Option<HeldOut>,
}

/// One removal step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Removal {
    pub candidates: usize,
    pub removed: usize,
    /// The groups removed as having no path to the output, which `removed` counts.
    pub dead: usize,
    /// `F` before and after, in bits.
    pub before_bits: f64,
    pub after_bits: f64,
    /// Every evaluated prefix `(k, F − F_before)` in bits, and every tested single group
    /// `(group, F − F_before)`.
    pub evaluations: Vec<(usize, f64)>,
    pub singles: Vec<(usize, f64)>,
    pub outcome: Outcome,
}

/// What a removal round accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    /// Groups with no path to the output, and possibly a prefix after them.
    Dead,
    /// A prefix of the groups in increasing estimated effect.
    Prefix,
    /// One group alone, from the single-group tests.
    Single,
    /// Nothing: the round's search (prefix bisection, then single groups) found no removal that
    /// does not increase `F`. The fit stops here; that is the search exhausted, not a proof.
    Exhausted,
}

/// The single groups a round tests when no prefix is accepted: this many of lowest estimated
/// effect and this many drawn at random (a search budget; acceptance is by `F` alone).
pub const SINGLES: usize = 32;

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub settings: Settings,
    pub training_sequences: usize,
    /// The scored tokens of every training experiment, `N`.
    pub scored_tokens: usize,
    /// The realized count of each experiment family in the training collection
    /// (`interchange::census`).
    pub families: BTreeMap<String, usize>,
    pub groups: usize,
    pub parameters: usize,
    /// The held-out evaluation at the starting point, on the fixed subset, and of the finished fit,
    /// on every held-out sequence.
    pub start: HeldOut,
    pub end: HeldOut,
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

    /// Per base one clean and one patched experiment over `variables` in `blocks` blocks
    /// (`interchange::sample`), from the batch's seed.
    fn experiments(&self, sequences: &[Vec<u32>], variables: &[ReadVariable], blocks: usize) -> Result<Vec<Experiment>, String> {
        let length = self.bases.first().map_or(0, |b| sequences[*b].len());
        interchange::sample(&mut StdRng::seed_from_u64(self.seed), self.bases.len(), variables, blocks, length)
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
    /// its first input, and that input. A shared block adds further terms (`library_sharing`).
    fn of(flat: &OperatorProgram, name: &str) -> Result<(Self, usize), String> {
        let (operator, bias) = (index_of(flat, name)?, operator_named(flat, &format!("{name}_bias")));
        let found = flat.nodes.iter().enumerate().find_map(|(n, node)| match node {
            Node::Affine { terms, bias: b } if *b == bias && terms.first().is_some_and(|t| t.1 == operator) => Some((n, terms[0].0)),
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
    /// The patch directions of each kept batch of experiments (by key), with its key under the
    /// protocol: fixed data, decomposed once.
    designs: BTreeMap<String, (u64, Arc<interchange::Design>)>,
    /// The fixed questions (`interchange::Protocol`): `M`'s read variables and directions, from
    /// the native program alone, whatever explanation this scorer scores.
    protocol: interchange::Protocol,
    /// Whether the explanation's own read variables align with the protocol's one for one (same
    /// order, blocks and rows), so that the adaptive family can ask them; tied and shared functions
    /// keep the variable of the call site they replace.
    aligned: bool,
    /// The directory of `M`'s target shards, when the fit keeps them.
    shards: Option<PathBuf>,
}

impl Scorer {
    fn new(device: &Device, native: &OperatorProgram, explanation: &Explanation, settings: &Settings, export: &str, shards: Option<PathBuf>) -> Result<Self, String> {
        let sites: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::library_reads(&explanation.artifact.program, sites.len())?;
        let protocol = interchange::Protocol::new(native, &sites, export)?;
        let aligned = reads.len() == protocol.variables().len()
            && reads.iter().zip(protocol.variables()).all(|(own, native)| {
                own.block == native.block && own.parts.len() == native.parts.len() && own.parts.iter().zip(&native.parts).all(|((_, rows), (_, native_rows))| rows.len() == native_rows.len())
            });
        let experiments =
            Interchange::new(device, native, &sites, &explanation.artifact, &explanation.trainable, reads, settings.numeric_bytes, settings.head_tile_rows)?;
        if let Some(dir) = &shards {
            std::fs::create_dir_all(dir).map_err(error)?;
        }
        let (flat, _, _) = interchange::sites(&explanation.artifact, &sites)?;
        let mlps = (0..sites.len()).map(|l| Mlp::of(&flat, l)).collect::<Result<_, _>>()?;
        let position = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        Ok(Self { experiments, mlps, position, designs: BTreeMap::new(), protocol, aligned, shards })
    }

    fn layers(&self) -> usize {
        self.mlps.len()
    }

    fn at(&self, operator: usize) -> Result<usize, String> {
        self.position.get(&operator).copied().ok_or_else(|| format!("operator {operator} is not trainable"))
    }

    /// The batch's experiments from `draw`: the fixed collection's for that batch.
    fn experiments(&self, draw: &Draw, sequences: &[Vec<u32>]) -> Result<Vec<Experiment>, String> {
        draw.experiments(sequences, self.protocol.variables(), 2 * self.layers())
    }

    /// `KL(M_e ‖ P_e)` per scored token in bits for `experiments` on `batch` at the explanation's
    /// `theta`, with the fixed directions (`M`'s reads), and with `gradient` the gradient of its
    /// sum in every trainable operator. `M`'s targets come from the shard `key` when the fit keeps
    /// shards and it holds them for these experiments, else are made (and kept there).
    fn score(&mut self, batch: &Batch, experiments: &[Experiment], theta: &[Array2<f64>], key: &str, gradient: bool) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>), String> {
        let (design, targets) = self.targets(batch, experiments, key)?;
        self.evaluated(batch, experiments, theta, &design, &targets, gradient)
    }

    /// [`Scorer::score`] with the posterior on the device: the explanation at its weight sample
    /// of `sample` ([`DevicePosterior::sample_into`]), or at its mean when none, and with
    /// `gradient` the gradient of the sum per trainable operator, left on the device.
    fn score_device(
        &mut self,
        posterior: &DevicePosterior,
        batch: &Batch,
        experiments: &[Experiment],
        sample: Option<u64>,
        key: &str,
        gradient: bool,
    ) -> Result<(Vec<Vec<f64>>, BTreeMap<usize, Tensor>), String> {
        let (design, targets) = self.targets(batch, experiments, key)?;
        match sample {
            Some(seed) => posterior.sample_into(self.experiments.explanation_mut(), seed)?,
            None => posterior.mean_into(self.experiments.explanation_mut())?,
        }
        let evaluation = self.experiments.evaluate_resident(batch, experiments, &design, &targets, gradient)?;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err("nonfinite explanation divergence".into());
        }
        Ok((evaluation.bits, evaluation.gradient))
    }

    /// The patch directions of `experiments` and `M`'s targets on them, from the shard `key` when
    /// the fit keeps them.
    fn targets(&mut self, batch: &Batch, experiments: &[Experiment], key: &str) -> Result<(Arc<interchange::Design>, Targets), String> {
        let fingerprint = self.protocol.key(batch, experiments);
        let design = match self.designs.get(key).filter(|(f, _)| *f == fingerprint) {
            Some((_, design)) => Arc::clone(design),
            None => {
                let design = Arc::new(self.protocol.design(self.experiments.models().0.program.device(), experiments)?);
                self.designs.insert(key.to_string(), (fingerprint, Arc::clone(&design)));
                design
            }
        };
        let path = self.shards.as_ref().map(|dir| dir.join(format!("{key}.bin")));
        let kept = match &path {
            Some(path) if path.exists() => Targets::read(self.experiments.models().0.program.device(), self.experiments.head(), path, fingerprint)?,
            _ => None,
        };
        let targets = match kept {
            Some(targets) => targets,
            None => {
                let targets = self.experiments.targets(batch, experiments, &design)?;
                if let Some(path) = &path {
                    targets.write(self.experiments.models().0.program.device(), path, fingerprint)?;
                }
                targets
            }
        };
        Ok((design, targets))
    }

    /// The patched `experiments`' divergence per scored token in bits at the explanation's `mean`,
    /// with the directions of the explanation's own reads there (the variables aligned with the
    /// protocol's): the adaptive family; none when the explanation's variables do not align.
    fn score_adaptive(&mut self, batch: &Batch, experiments: &[Experiment], mean: &[Array2<f64>]) -> Result<Option<Vec<Vec<f64>>>, String> {
        if !self.aligned {
            return Ok(None);
        }
        let own = self.experiments.variables().to_vec();
        let design = self.experiments.design_at(&own, experiments, mean)?;
        let targets = self.experiments.targets(batch, experiments, &design)?;
        Ok(Some(self.evaluated(batch, experiments, mean, &design, &targets, false)?.0))
    }

    fn evaluated(
        &mut self,
        batch: &Batch,
        experiments: &[Experiment],
        theta: &[Array2<f64>],
        design: &interchange::Design,
        targets: &Targets,
        gradient: bool,
    ) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>), String> {
        self.experiments.load(theta)?;
        let evaluation = self.experiments.evaluate_resident(batch, experiments, design, targets, gradient)?;
        let p = self.experiments.models().1;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err("nonfinite explanation divergence".into());
        }
        if !gradient {
            return Ok((evaluation.bits, Vec::new()));
        }
        let d = p.program.device();
        let mut gradients = Vec::with_capacity(theta.len());
        for (op, values) in self.position.iter().map(|(op, i)| (*op, &theta[*i])) {
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

/// The held-out batches of `sequences` and their experiments: a fixed sample from the training
/// distribution, one clean and one patched experiment per base, drawn from each batch's seed.
fn held_out_experiments(scorer: &Scorer, sequences: &[Vec<u32>], settings: &Settings) -> Result<Vec<(Draw, Vec<Experiment>)>, String> {
    draws(sequences.len(), settings.batch_sequences, settings.seed)?
        .into_iter()
        .map(|draw| {
            let experiments = scorer.experiments(&draw, sequences)?;
            Ok((draw, experiments))
        })
        .collect()
}

/// The held-out evaluation of `posterior` (held on the host, and as `device_posterior` on the
/// device) on `sequences` (module note); `tokens` is `N`.
#[allow(clippy::too_many_arguments)]
fn held_out(
    scorer: &mut Scorer,
    explanation: &Explanation,
    posterior: &Posterior,
    device_posterior: &DevicePosterior,
    sequences: &[Vec<u32>],
    settings: &Settings,
    tokens: usize,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<HeldOut, String> {
    let blocks = 2 * scorer.layers();
    let (mut clean, mut patched) = (vec![Mean::default(); blocks], vec![Mean::default(); blocks]);
    let (mut read, mut joint, mut sampled, mut adaptive) = (Mean::default(), Mean::default(), Mean::default(), Mean::default());
    let size = |e: &Experiment| e.explained.iter().filter(|x| **x).count();
    for (b, (draw, experiments)) in held_out_experiments(scorer, sequences, settings)?.into_iter().enumerate() {
        let batch = draw.batch(sequences)?;
        let key = format!("held_{}_{b}", sequences.len());
        let (bits, _) = scorer.score_device(device_posterior, &batch, &experiments, None, &key, false)?;
        for (e, bits) in experiments.iter().zip(&bits) {
            match &e.patch {
                None => clean[size(e) - 1].add(bits),
                Some(patch) => {
                    patched[size(e) - 1].add(bits);
                    match patch {
                        Patch::Read { .. } => read.add(bits),
                        Patch::Reads { .. } | Patch::Complement { .. } => joint.add(bits),
                    }
                }
            }
        }
        let (bits, _) = scorer.score_device(device_posterior, &batch, &experiments, Some(noise_seed(settings.seed, 0, b)), &key, false)?;
        bits.iter().for_each(|b| sampled.add(b));
        let patched: Vec<Experiment> = experiments.into_iter().filter(|e| e.patch.is_some()).collect();
        if let Some(bits) = scorer.score_adaptive(&batch, &patched, &posterior.mean)? {
            bits.iter().for_each(|b| adaptive.add(b));
        }
    }
    let data = sampled.mean().ok_or("no held-out tokens")?;
    let divergence: f64 = posterior.divergences().iter().sum();
    let gaussian = posterior.description();
    let prior_nats = match prior {
        Some(prior) => prior_term(prior, posterior, noise_seed(settings.seed, 0, 0), false)?.0,
        None => 0.0,
    };
    let description = gaussian + explanation.fixed_nats + prior_nats;
    Ok(HeldOut {
        objective_bits_per_token: data + description / LN_2 / tokens as f64,
        data_bits_per_token: data,
        divergence_bits: divergence / LN_2,
        variance_bits: (gaussian - divergence) / LN_2,
        choice_bits: explanation.fixed_nats / LN_2,
        prior_bits: prior_nats / LN_2,
        clean: clean.iter().map(Mean::mean).collect(),
        patched: patched.iter().map(Mean::mean).collect(),
        read_patch: read.mean(),
        joint_patch: joint.mean(),
        adaptive_patch: adaptive.mean(),
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
    identity: Identity,
    settings: Settings,
    tokens: usize,
    shapes: Vec<(usize, usize)>,
    /// The held-out evaluation at the starting point, on the fixed subset.
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
    /// The training and the per-epoch evaluation seconds so far, and the last full held-out
    /// evaluation's seconds (the schedule's inputs).
    #[serde(default)]
    training_seconds: f64,
    #[serde(default)]
    evaluation_seconds: f64,
    #[serde(default)]
    full_seconds: f64,
    /// The prior term's state, when the fit has one.
    #[serde(default)]
    prior: Option<serde_json::Value>,
}

/// What a checkpoint belongs to: a fit resumes from it only when every field agrees, so a
/// checkpoint of another model, dataset or library can neither resume nor skip training. Each
/// field is a SHA-256 in hexadecimal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    /// The export the native model was imported from (`export.json`'s digest).
    pub export: String,
    /// The training sequences' token ids, then the held-out sequences', in order.
    pub tokens: String,
    /// The structure of the native program and of the explanation's program: their nodes,
    /// rules, operators' names and interfaces, and outputs (the values are the export's).
    pub program: String,
    /// The prior groups: every group's name and member entries.
    pub groups: String,
    /// The shared-parameter map: per trainable operator, every node of every rule and of the
    /// program that applies it.
    pub sharing: String,
}

fn program_structure(hasher: &mut Fingerprinter, tag: &[u8], program: &OperatorProgram) {
    hasher.absorb_tag(tag);
    for node in &program.nodes {
        hasher.absorb_str(b"node", &format!("{node:?}"));
    }
    for rule in &program.rules {
        hasher.absorb_str(b"rule", &format!("{}|{:?}|{:?}|{}", rule.name, rule.inputs, rule.nodes, rule.output));
    }
    for op in &program.operators {
        hasher.absorb_str(b"operator", &format!("{}|{:?}|{:?}", op.name, op.rows, op.cols));
    }
    hasher.absorb_u64(b"output", program.output as u64);
}

/// The identity of a fit of `explanation` (made from `native`, imported from the export whose
/// digest is `export`) on the training `sequences` with the `held` sequences held out.
pub fn identity(export: &str, native: &OperatorProgram, explanation: &Explanation, sequences: &[Vec<u32>], held: &[Vec<u32>]) -> Identity {
    let mut tokens = Fingerprinter::new();
    for (tag, set) in [(&b"training"[..], sequences), (&b"held out"[..], held)] {
        tokens.absorb_u64(tag, set.len() as u64);
        for sequence in set {
            let bytes: Vec<u8> = sequence.iter().flat_map(|t| t.to_le_bytes()).collect();
            tokens.absorb_bytes(b"sequence", &bytes);
        }
    }
    let mut program = Fingerprinter::new();
    program_structure(&mut program, b"native", native);
    program_structure(&mut program, b"explanation", &explanation.artifact.program);
    let mut groups = Fingerprinter::new();
    groups.absorb_str(b"removed", &format!("{:?}", explanation.removed));
    groups.absorb_u64(b"fixed", explanation.fixed_nats.to_bits());
    for group in &explanation.groups {
        groups.absorb_str(b"group", &group.name);
        for cell in &group.cells {
            groups.absorb_str(b"cells", &format!("{}|{:?}|{:?}", cell.operator, cell.rows, cell.cols));
        }
    }
    let mut sharing = Fingerprinter::new();
    let explained = &explanation.artifact.program;
    let bodies = std::iter::once(("program".to_string(), &explained.nodes)).chain(explained.rules.iter().map(|r| (r.name.clone(), &r.nodes)));
    let mut uses: BTreeMap<usize, Vec<String>> = explanation.trainable.iter().map(|op| (*op, Vec::new())).collect();
    for (body, nodes) in bodies {
        for (n, node) in nodes.iter().enumerate() {
            for op in node.operators() {
                if let Some(list) = uses.get_mut(&op) {
                    list.push(format!("{body}:{n}"));
                }
            }
        }
    }
    for (op, list) in &uses {
        sharing.absorb_str(b"uses", &format!("{op}|{}", list.join(",")));
    }
    Identity {
        export: export.to_string(),
        tokens: tokens.finalize().to_hex(),
        program: program.finalize().to_hex(),
        groups: groups.finalize().to_hex(),
        sharing: sharing.finalize().to_hex(),
    }
}

/// Read only the length-delimited JSON header. Neither identity checks nor restoration need a
/// second, checkpoint-sized byte buffer beside the posterior and its optimizer state.
fn checkpoint_header<T: serde::de::DeserializeOwned>(path: &Path) -> Result<(T, BufReader<std::fs::File>, u64), String> {
    let file = std::fs::File::open(path).map_err(error)?;
    let file_bytes = file.metadata().map_err(error)?.len();
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut prefix = [0_u8; 8];
    reader
        .read_exact(&mut prefix)
        .map_err(|e| if e.kind() == std::io::ErrorKind::UnexpectedEof { "a truncated checkpoint".to_string() } else { error(e) })?;
    let header_bytes = u64::from_le_bytes(prefix);
    let payload_start = header_bytes.checked_add(8).filter(|end| *end <= file_bytes).ok_or("a truncated checkpoint")?;
    // `take` prevents a malformed JSON header from consuming binary coefficient bytes. JSON
    // decoding allocates its actual fields, not the advertised header or payload byte count.
    let header = serde_json::from_reader((&mut reader).take(header_bytes)).map_err(error)?;
    Ok((header, reader, file_bytes - payload_start))
}

/// The checkpoint at `path`'s identity.
fn checkpoint_identity(path: &Path) -> Result<Identity, String> {
    #[derive(Deserialize)]
    struct Header {
        identity: Identity,
    }
    let (header, _, _): (Header, _, _) =
        checkpoint_header(path).map_err(|e| format!("{}: a checkpoint without this fit's identity: {e}", path.display()))?;
    Ok(header.identity)
}

/// Refuse a checkpoint at `path` of another fit than `identity`'s, naming every field that
/// differs; a path with no checkpoint passes. A driver calls this before it writes anything.
pub fn check_checkpoint(path: &Path, identity: &Identity) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let found = checkpoint_identity(path)?;
    check_checkpoint_identity(path, &found, identity)
}

fn check_checkpoint_identity(path: &Path, found: &Identity, identity: &Identity) -> Result<(), String> {
    let differing: Vec<&str> = [
        ("export", found.export == identity.export),
        ("tokens", found.tokens == identity.tokens),
        ("program", found.program == identity.program),
        ("groups", found.groups == identity.groups),
        ("sharing", found.sharing == identity.sharing),
    ]
    .into_iter()
    .filter(|(_, same)| !same)
    .map(|(field, _)| field)
    .collect();
    if differing.is_empty() {
        Ok(())
    } else {
        Err(format!("{}: a checkpoint of another fit (its {} differ)", path.display(), differing.join(", ")))
    }
}

/// The arrays a checkpoint holds per trainable operator: `μ`, `ln σ` and IVON's state (the
/// gradient's momentum and the curvature estimate), each as little-endian float64.
const CHECKPOINT_ARRAYS: usize = 4;

/// The payload bytes of a checkpoint of operators of `shapes`.
fn checkpoint_payload_bytes(shapes: &[(usize, usize)]) -> Option<u64> {
    shapes.iter().try_fold(0_u64, |total, &(rows, cols)| {
        let cells = u64::try_from(rows).ok()?.checked_mul(u64::try_from(cols).ok()?)?;
        total.checked_add(cells.checked_mul(CHECKPOINT_ARRAYS as u64 * 8)?)
    })
}

/// Decode in fixed-size byte tiles directly into the destination, including nonstandard array
/// layouts. The wire order remains ndarray's logical iteration order used by the writer.
fn read_checkpoint_array(reader: &mut impl Read, array: &mut Array2<f64>) -> Result<(), String> {
    let mut bytes = [0_u8; 64 * 1024];
    let mut remaining = array.len();
    let mut values = array.iter_mut();
    while remaining != 0 {
        let count = remaining.min(bytes.len() / 8);
        reader.read_exact(&mut bytes[..count * 8]).map_err(error)?;
        for (value, encoded) in values.by_ref().take(count).zip(bytes[..count * 8].chunks_exact(8)) {
            *value = f64::from_le_bytes(encoded.try_into().expect("eight bytes"));
        }
        remaining -= count;
    }
    Ok(())
}

/// Write the checkpoint atomically: the progress as JSON after its length, then every array's
/// values as little-endian float64. The progress alone also goes to the path with extension
/// `json`, readable while the fit runs.
fn save_checkpoint(path: &Path, progress: &Progress, posterior: &DevicePosterior) -> Result<(), String> {
    let header = serde_json::to_vec(progress).map_err(error)?;
    let partial = path.with_extension("partial");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&partial).map_err(error)?);
    file.write_all(&(header.len() as u64).to_le_bytes()).map_err(error)?;
    file.write_all(&header).map_err(error)?;
    // Per trainable operator `μ`, `ln σ` and IVON's state (the gradient's momentum and the
    // curvature estimate), one operator on the host at a time.
    for i in 0..progress.shapes.len() {
        let (mean, log_sd, moments) = posterior.operator(i)?;
        for array in [&mean, &log_sd].into_iter().chain(&moments) {
            for value in array.iter() {
                file.write_all(&value.to_le_bytes()).map_err(error)?;
            }
        }
    }
    file.into_inner().map_err(error)?.sync_all().map_err(error)?;
    std::fs::rename(&partial, path).map_err(error)?;
    let partial = path.with_extension("json.partial");
    std::fs::write(&partial, serde_json::to_vec_pretty(progress).map_err(error)?).map_err(error)?;
    std::fs::rename(&partial, path.with_extension("json")).map_err(error)
}

/// Restore a checkpoint of this fit into `posterior`, with IVON's state per operator, or refuse one
/// of another fit.
fn load_checkpoint(path: &Path, expected: &Progress, posterior: &mut Posterior) -> Result<(Progress, Vec<[Array2<f64>; 2]>), String> {
    let (progress, mut reader, payload_bytes): (Progress, _, _) = checkpoint_header(path)?;
    check_checkpoint_identity(path, &progress.identity, &expected.identity)?;
    let same_settings =
        serde_json::to_value(&progress.settings).map_err(error)? == serde_json::to_value(&expected.settings).map_err(error)?;
    if !same_settings
        || progress.tokens != expected.tokens
        || progress.shapes != expected.shapes
        || progress.active.len() != expected.active.len()
    {
        return Err(format!("{}: a checkpoint of another fit", path.display()));
    }
    if checkpoint_payload_bytes(&progress.shapes) != Some(payload_bytes)
        || posterior.mean.len() != progress.shapes.len()
        || posterior.log_sd.len() != progress.shapes.len()
        || posterior.mean.iter().zip(&progress.shapes).any(|(array, shape)| array.dim() != *shape)
        || posterior.log_sd.iter().zip(&progress.shapes).any(|(array, shape)| array.dim() != *shape)
    {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    let mut moments = Vec::with_capacity(posterior.mean.len());
    for i in 0..posterior.mean.len() {
        let dim = posterior.mean[i].dim();
        read_checkpoint_array(&mut reader, &mut posterior.mean[i])?;
        read_checkpoint_array(&mut reader, &mut posterior.log_sd[i])?;
        let mut next = std::array::from_fn(|_| Array2::zeros(dim));
        for array in &mut next {
            read_checkpoint_array(&mut reader, array)?;
        }
        moments.push(next);
    }
    if reader.read(&mut [0_u8; 1]).map_err(error)? != 0 {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    posterior.active = progress.active.clone();
    Ok((progress, moments))
}

/// The posterior of a fit checkpoint of `explanation` (`OUT/checkpoint.bin`, [`fit`]): its means,
/// log standard deviations and active groups, for reading a fit that is still running.
pub fn checkpoint_posterior(explanation: &Explanation, path: &Path) -> Result<Posterior, String> {
    #[derive(Deserialize)]
    struct Header {
        tokens: usize,
        shapes: Vec<(usize, usize)>,
        active: Vec<bool>,
    }
    let (header, mut reader, payload_bytes): (Header, _, _) = checkpoint_header(path)?;
    let mut posterior = Posterior::new(explanation, header.tokens)?;
    if header.shapes != posterior.mean.iter().map(Array2::dim).collect::<Vec<_>>() || header.active.len() != posterior.active.len() {
        return Err(format!("{}: a checkpoint of another explanation", path.display()));
    }
    if checkpoint_payload_bytes(&header.shapes) != Some(payload_bytes) {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    for i in 0..header.shapes.len() {
        read_checkpoint_array(&mut reader, &mut posterior.mean[i])?;
        read_checkpoint_array(&mut reader, &mut posterior.log_sd[i])?;
        let mut state = Array2::zeros(header.shapes[i]);
        for _ in 2..CHECKPOINT_ARRAYS {
            read_checkpoint_array(&mut reader, &mut state)?;
        }
    }
    posterior.active = header.active;
    Ok(posterior)
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
    export: &str,
    checkpoint: Option<&Path>,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<Fit, String> {
    settings.validate()?;
    let mut prior = prior;
    let length = sequences.first().map_or(0, Vec::len);
    if length == 0 || held.len() < 2 || sequences.iter().chain(held).any(|s| s.len() != length) {
        return Err("training and held-out sequences (at least two) must be nonempty and of one length".into());
    }
    let started = Instant::now();
    let draws = draws(sequences.len(), settings.batch_sequences, settings.seed)?;
    if draws.len() < 2 {
        return Err("the convergence test needs at least two training batches".into());
    }
    let shards = checkpoint.map(|path| path.with_extension("targets"));
    let mut scorer = Scorer::new(device, native, explanation, settings, export, shards)?;
    // The fixed collection: its scored tokens N and its realized families.
    let (mut tokens, mut families) = (0, BTreeMap::new());
    for draw in &draws {
        let experiments = scorer.experiments(draw, sequences)?;
        tokens += experiments.iter().map(|e| length - e.position).sum::<usize>();
        for (family, count) in interchange::census(&experiments, scorer.protocol.variables()) {
            *families.entry(family.to_string()).or_insert(0) += count;
        }
    }
    log::info!("library training collection: {tokens} scored tokens, families {families:?}");
    let mut posterior = Posterior::new(explanation, tokens)?;
    let parameters = posterior.mean.iter().map(Array2::len).sum();
    let mut progress = Progress {
        identity: identity(export, native, explanation, sequences, held),
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
        training_seconds: 0.0,
        evaluation_seconds: 0.0,
        full_seconds: 0.0,
        prior: None,
    };
    // The fixed held-out subset: the first batch of held-out bases (at least the two a source
    // needs).
    let subset = &held[..settings.batch_sequences.clamp(2, held.len())];
    let mut resumed = None;
    if let Some(path) = checkpoint.filter(|p| p.exists()) {
        let (loaded, moments) = load_checkpoint(path, &progress, &mut posterior)?;
        progress = loaded;
        match (prior.as_deref_mut(), &progress.prior) {
            (Some(prior), Some(state)) => prior.load(state)?,
            (None, None) => {}
            _ => return Err(format!("{}: a checkpoint of a fit with another prior term", path.display())),
        }
        resumed = Some(moments);
        log::info!("library fit resumed at epoch {} from {}", progress.epoch, path.display());
    }
    let resumed_seconds = progress.seconds;
    let mut device_posterior = DevicePosterior::new(device, explanation, &posterior, tokens as f64, resumed.as_deref(), u64::try_from(progress.step).map_err(error)?)?;
    drop(resumed);
    // The curvature estimate averages over one epoch's batches: each batch weighs about once.
    let ivon = Ivon { rate: settings.rate, beta1: settings.beta1, beta2: 1.0 - 1.0 / draws.len() as f64 };
    // Each group's size, whose `½ ln |G|` an active group's variance costs.
    let sizes: Vec<f64> = explanation.groups.iter().map(|g| g.cells.iter().map(|c| (c.rows.len() * c.cols.len()) as f64).sum()).collect();
    // After every save, the posterior-mean artifact goes next to the checkpoint (extension
    // `artifact.bin`), so the current explanation can be read and scored while the fit runs.
    let save = |progress: &mut Progress, posterior: &Posterior, device_posterior: &DevicePosterior| -> Result<(), String> {
        progress.active = posterior.active.clone();
        progress.seconds = resumed_seconds + started.elapsed().as_secs_f64();
        let Some(path) = checkpoint else { return Ok(()) };
        save_checkpoint(path, progress, device_posterior)?;
        let partial = path.with_extension("artifact.partial");
        std::fs::write(&partial, posterior_mean(explanation, posterior)?.f32_literals()?.to_bytes()?).map_err(error)?;
        std::fs::rename(&partial, path.with_extension("artifact.bin")).map_err(error)
    };
    if let Some(prior) = prior.as_deref_mut()
        && progress.prior.is_none()
    {
        prior.epoch(explanation, &posterior)?;
    }
    if progress.start.is_none() {
        // The start obeys the budget too: the subset now; the full set's time, until a full
        // evaluation is made, estimated from the subset's in proportion to the sequences.
        let timed = Instant::now();
        let start = held_out(&mut scorer, explanation, &posterior, &device_posterior, subset, settings, tokens, prior.as_deref_mut())?;
        let seconds = timed.elapsed().as_secs_f64();
        progress.evaluation_seconds += seconds;
        progress.full_seconds = seconds * held.len() as f64 / subset.len() as f64;
        log::info!("library start: {start:?}");
        progress.start = Some(start);
        device_posterior.values_into(&mut posterior)?;
        progress.prior = prior.as_deref().map(PriorTerm::save).transpose()?;
        save(&mut progress, &posterior, &device_posterior)?;
    }
    while !progress.done {
        let epoch = progress.epoch;
        let epoch_started = Instant::now();
        if let Some(prior) = prior.as_deref_mut()
            && epoch > 0
        {
            prior.epoch(explanation, &posterior)?;
        }
        // Selection may change the required operators each epoch. Only these means and
        // deviations cross to the host per step for the current CPU prior implementation.
        let prior_operators = prior.as_deref().map(PriorTerm::operators).unwrap_or_default();
        // Which groups are active changes only at a removal.
        let subset_code = posterior.subset_nats();
        let mut estimates = Vec::with_capacity(draws.len());
        let (mut data_sum, mut description_sum) = (0.0, 0.0);
        let (mut clean, mut patched) = (Mean::default(), Mean::default());
        for (b, draw) in draws.iter().enumerate() {
            let step_started = Instant::now();
            let batch = draw.batch(sequences)?;
            let experiments = scorer.experiments(draw, sequences)?;
            let key = noise_seed(settings.seed, epoch + 1, b);
            let (bits, mut gradients) = scorer.score_device(&device_posterior, &batch, &experiments, Some(key), &format!("train_{b}"), true)?;
            for (e, bits) in experiments.iter().zip(&bits) {
                if e.patch.is_some() { patched.add(bits) } else { clean.add(bits) }
            }
            let scored = bits.iter().map(Vec::len).sum::<usize>();
            let scale = tokens as f64 / scored as f64;
            let data = scale * LN_2 * bits.iter().flatten().sum::<f64>();
            // `Σ_G KL_G` and the active groups' variances at the posterior the sample was drawn from.
            let variances = device_posterior.variances()?;
            let description: f64 = device_posterior
                .divergences()?
                .iter()
                .zip(&sizes)
                .zip(&posterior.active)
                .zip(variances.iter().zip(&posterior.initial))
                .map(|(((d, n), active), (v, initial))| if *active { d + 0.5 * n.ln() + scale_bits(*v, *initial) * LN_2 } else { 0.0 })
                .sum::<f64>()
                + subset_code;
            // The prior term at the same weight sample; its gradient joins the data term's, which
            // the step weighs by `scale` in nats.
            let prior_nats = match prior.as_deref_mut() {
                Some(prior) => {
                    for &i in &prior_operators {
                        let (mean, log_sd) = device_posterior.values(i)?;
                        posterior.mean[i] = mean;
                        posterior.log_sd[i] = log_sd;
                    }
                    let (nats, extra) = prior_term(prior, &posterior, key, true)?;
                    for (i, g) in extra {
                        let op = explanation.trainable[i];
                        let uploaded = device.upload((g / (scale * LN_2)).view()).map_err(error)?;
                        match gradients.get_mut(&op) {
                            Some(total) => device.axpy(total, 1.0, &uploaded).map_err(error)?,
                            None => {
                                gradients.insert(op, uploaded);
                            }
                        }
                    }
                    nats
                }
                None => 0.0,
            };
            let description = description + explanation.fixed_nats + prior_nats;
            if !description.is_finite() {
                return Err("a nonfinite posterior divergence".into());
            }
            estimates.push(data + description);
            data_sum += data;
            description_sum += description;
            progress.step += 1;
            device_posterior.step(&gradients, LN_2 / scored as f64, &ivon, key)?;
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
                progress.training_seconds += epoch_started.elapsed().as_secs_f64();
                device_posterior.values_into(&mut posterior)?;
                let timed = Instant::now();
                let evaluation = held_out(&mut scorer, explanation, &posterior, &device_posterior, subset, settings, tokens, prior.as_deref_mut())?;
                progress.evaluation_seconds += timed.elapsed().as_secs_f64();
                evaluation
            },
            held_out_full: if progress.evaluation_seconds + progress.full_seconds <= EVALUATION_SHARE * progress.training_seconds {
                let timed = Instant::now();
                let evaluation = held_out(&mut scorer, explanation, &posterior, &device_posterior, held, settings, tokens, prior.as_deref_mut())?;
                progress.full_seconds = timed.elapsed().as_secs_f64();
                progress.evaluation_seconds += progress.full_seconds;
                Some(evaluation)
            } else {
                None
            },
        };
        log::info!("library fit epoch {epoch}: {record:?}");
        progress.epochs.push(record);
        progress.previous = Some(estimates);
        progress.epoch += 1;
        let converged = matches!((improvement, standard_error), (Some(i), Some(se)) if i <= se);
        if converged {
            let round = progress.removals.len();
            let removal = remove(&mut scorer, &mut posterior, &draws, sequences, settings, explanation, round, prior.as_deref_mut())?;
            log::info!("library removal after epoch {epoch}: {} of {} candidates ({:?})", removal.removed, removal.candidates, removal.outcome);
            // The removed groups' entries are exactly zero with `ln σ = −∞`, which the device step
            // leaves alone.
            device_posterior.set_values(&posterior)?;
            progress.done = removal.removed == 0;
            progress.removals.push(removal);
            // The objective changed discretely: convergence is judged afresh.
            progress.previous = None;
        }
        progress.prior = prior.as_deref().map(PriorTerm::save).transpose()?;
        save(&mut progress, &posterior, &device_posterior)?;
    }
    let objective_bits = progress.removals.last().map_or(f64::NAN, |r| r.after_bits);
    let end = held_out(&mut scorer, explanation, &posterior, &device_posterior, held, settings, tokens, prior.as_deref_mut())?;
    log::info!("library end: {end:?}");
    Ok(Fit {
        report: Report {
            settings: settings.clone(),
            training_sequences: sequences.len(),
            scored_tokens: tokens,
            families,
            groups: explanation.groups.len(),
            parameters,
            start: progress.start.ok_or("no starting evaluation")?,
            end,
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
/// with the groups `removed` (and the already removed ones) zeroed, on the fixed collection; with
/// `prior`, plus its value at each batch's sample, averaged, and the parameters it sends.
fn expected_divergence(
    scorer: &mut Scorer,
    posterior: &Posterior,
    draws: &[Draw],
    sequences: &[Vec<u32>],
    removed: &[usize],
    settings: &Settings,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<f64, String> {
    let mut trial = posterior.clone();
    trial.remove(removed);
    let mut prior = prior;
    let (mut bits, mut prior_nats) = (0.0, 0.0);
    for (b, draw) in draws.iter().enumerate() {
        // Removal zeroes entries, so the remaining entries see the same noise as the full posterior.
        let (theta, _) = trial.sample(noise_seed(settings.seed, 0, b));
        let experiments = scorer.experiments(draw, sequences)?;
        let (scored, _) = scorer.score(&draw.batch(sequences)?, &experiments, &theta, &format!("train_{b}"), false)?;
        bits += scored.iter().flatten().sum::<f64>();
        if let Some(prior) = prior.as_deref_mut() {
            let sample: BTreeMap<usize, Array2<f64>> = prior.operators().into_iter().map(|i| (i, theta[i].clone())).collect();
            prior_nats += prior.sample(&trial, &sample, false)?.0;
        }
    }
    let cost = prior.as_deref().map_or(Ok(0.0), |p| p.cost(&trial))?;
    Ok(bits * LN_2 + prior_nats / draws.len() as f64 + cost)
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

/// The groups with no path to the output while `active` holds (module note): the planes of a head
/// none of whose value coordinates is active, and the other groups of an MLP function one of whose
/// groups is inactive. An output that a read–write tie replaces (`Explanation::removed`) is another
/// path, so with ties a function's removed output does not make its gate dead. Removing these
/// groups removes no other group's only path, so one pass reaches the fixed point.
fn dead_groups(explanation: &Explanation, active: &[bool]) -> Vec<usize> {
    let tied = !explanation.removed.is_empty();
    let mut dead = std::collections::BTreeSet::new();
    for layer in &explanation.layers {
        for (planes, values) in &layer.heads {
            if !values.iter().any(|g| active[*g]) {
                dead.extend(planes.iter().copied().filter(|g| active[*g]));
            }
        }
        for groups in &layer.functions {
            let Some((&output, reads)) = groups.split_last() else { continue };
            let read_removed = reads.iter().any(|g| !active[*g]);
            // A function with a read removed writes zero, so its output is dead; one that writes
            // nothing (a read or its output removed) leaves its reads dead, unless a read may be
            // another function's write (a tie).
            if read_removed && active[output] {
                dead.insert(output);
            }
            if (read_removed || !active[output]) && !tied {
                dead.extend(reads.iter().copied().filter(|g| active[*g]));
            }
        }
    }
    dead.into_iter().collect()
}

/// A removal round (module note): the dead groups, then prefixes of the rest in increasing
/// estimated effect by bisection, then, when neither is accepted, single groups; every proposal
/// compensated in the MLPs it deletes functions of (`library_compensation`). `round` seeds the
/// random single groups.
fn remove(
    scorer: &mut Scorer,
    posterior: &mut Posterior,
    draws: &[Draw],
    sequences: &[Vec<u32>],
    settings: &Settings,
    explanation: &Explanation,
    round: usize,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<Removal, String> {
    let mut prior = prior;
    let fixed = explanation.fixed_nats;
    let to_bits = |nats: f64| nats / LN_2;
    let candidates = posterior.active.iter().filter(|a| **a).count();
    let before = expected_divergence(scorer, posterior, draws, sequences, &[], settings, prior.as_deref_mut())? + posterior.description() + fixed;
    // The compensated posterior with `groups` removed and its `F` minus `base`.
    let trial = |scorer: &mut Scorer, compensation: &Compensation, posterior: &Posterior, groups: &[usize], base: f64, prior: Option<&mut (dyn PriorTerm + 'static)>| -> Result<(Posterior, f64), String> {
        let proposal = compensation.proposal(posterior, groups)?;
        let c = expected_divergence(scorer, &proposal, draws, sequences, &[], settings, prior)? + proposal.description() + fixed - base;
        if !c.is_finite() {
            return Err("nonfinite group removal objective".into());
        }
        Ok((proposal, c))
    };
    // The dead groups, all at once.
    let mut base = before;
    let mut compensation = Compensation::new(&mut scorer.experiments, explanation, posterior, sequences, settings.batch_sequences)?;
    let dead = dead_groups(explanation, &posterior.active);
    let mut removed_dead = 0;
    if !dead.is_empty() {
        let (proposal, c) = trial(scorer, &compensation, posterior, &dead, base, prior.as_deref_mut())?;
        log::info!("library removal of {} dead groups: F changes by {:.6e} bits", dead.len(), to_bits(c));
        if c <= 0.0 {
            *posterior = proposal;
            base += c;
            removed_dead = dead.len();
            compensation = Compensation::new(&mut scorer.experiments, explanation, posterior, sequences, settings.batch_sequences)?;
        }
    }
    // Prefixes in increasing estimated effect.
    let estimates = posterior.removal_estimates();
    let mut order: Vec<usize> = (0..estimates.len()).filter(|g| posterior.active[*g]).collect();
    order.sort_by(|a, b| estimates[*a].total_cmp(&estimates[*b]));
    let mut evaluations: Vec<(usize, f64)> = Vec::new();
    let low = {
        let current: &Posterior = posterior;
        let mut change = |k: usize| -> Result<f64, String> {
            if let Some((_, c)) = evaluations.iter().find(|(at, _)| *at == k) {
                return Ok(*c);
            }
            let (_, c) = trial(scorer, &compensation, current, &order[..k], base, prior.as_deref_mut())?;
            log::info!("library removal of {k} of {} groups: F changes by {:.6e} bits", order.len(), to_bits(c));
            evaluations.push((k, c));
            Ok(c)
        };
        largest_accepted_prefix(order.len(), &mut change)?
    };
    let mut singles: Vec<(usize, f64)> = Vec::new();
    let outcome = if low > 0 {
        let (proposal, c) = trial(scorer, &compensation, posterior, &order[..low], base, prior.as_deref_mut())?;
        *posterior = proposal;
        base += c;
        if removed_dead > 0 { Outcome::Dead } else { Outcome::Prefix }
    } else if removed_dead > 0 {
        Outcome::Dead
    } else {
        // Single groups: the lowest estimates, then random others.
        let mut tested: Vec<usize> = order.iter().copied().take(SINGLES).collect();
        let mut rest: Vec<usize> = order.iter().copied().skip(SINGLES).collect();
        let mut rng = StdRng::seed_from_u64(noise_seed(settings.seed, round, usize::MAX));
        for _ in 0..SINGLES.min(rest.len()) {
            let at = rng.random_range(0..rest.len());
            tested.push(rest.swap_remove(at));
        }
        let mut best: Option<(Posterior, usize, f64)> = None;
        for &g in &tested {
            let (proposal, c) = trial(scorer, &compensation, posterior, &[g], base, prior.as_deref_mut())?;
            singles.push((g, c));
            if c <= 0.0 && best.as_ref().is_none_or(|(_, _, b)| c < *b) {
                best = Some((proposal, g, c));
            }
        }
        match best {
            Some((proposal, g, c)) => {
                log::info!("library removal of single group {g}: F changes by {:.6e} bits", to_bits(c));
                *posterior = proposal;
                base += c;
                Outcome::Single
            }
            None => Outcome::Exhausted,
        }
    };
    let removed = candidates - posterior.active.iter().filter(|a| **a).count();
    Ok(Removal {
        candidates,
        removed,
        dead: removed_dead,
        before_bits: to_bits(before),
        after_bits: to_bits(base),
        evaluations: evaluations.into_iter().map(|(k, c)| (k, to_bits(c))).collect(),
        singles: singles.into_iter().map(|(g, c)| (g, to_bits(c))).collect(),
        outcome,
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
    fn compact_targets_of_a_tied_qwen3_head_are_the_full_logit_divergence() {
        use crate::resident_causal_fit::fixed_head_target::{Head, ResidentHead, Teacher};
        use gam_gpu::tensor::Arithmetic;
        // M's head is its token embedding read transposed (tied) after the final norm. `P` is M
        // with the first MLP's output map scaled, so the two next-token distributions differ.
        let (native, _, family, _) = tiny_qwen3("library_tied_head");
        let mut changed = native.clone();
        let down = changed.operators.iter().position(|op| op.name == "blocks.0.down_proj").unwrap();
        let op = Arc::clone(&changed.operators[down]);
        changed.operators[down] = Arc::new(library_operator(&op.name, op.rows.clone(), op.cols.clone(), op.matrix() * 1.7, &op.name).unwrap());
        let logits = |program: &OperatorProgram| program.execute(&family, false).unwrap().values[program.output].clone();
        let (p, q) = (logits(&native), logits(&changed));
        let log_softmax = |row: ndarray::ArrayView1<'_, f64>| {
            let peak = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let partition = peak + row.iter().map(|x| (x - peak).exp()).sum::<f64>().ln();
            row.mapv(|x| x - partition)
        };
        let full: Vec<f64> = p
            .rows()
            .into_iter()
            .zip(q.rows())
            .map(|(p, q)| {
                let (lp, lq) = (log_softmax(p), log_softmax(q));
                lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
            })
            .collect();
        let device = Device::host();
        let head = Head::of(&native).unwrap();
        let target = Teacher::new(&device, &native, 16, 1 << 26).unwrap().target(&family, None).unwrap();
        let prefix = crate::device_program::DeviceProgram::compile_values(&device, &head.prefix(&changed)).unwrap();
        let trace = prefix.forward(&family).unwrap();
        let (compact, _) =
            ResidentHead::new(&device, &head, 16).unwrap().score(&device, trace.value(prefix.hidden()).unwrap(), &target, false, Arithmetic::F64).unwrap();
        assert_eq!(compact.len(), full.len());
        assert!(full.iter().sum::<f64>() > 1e-3, "the scaled MLP moves the distributions");
        for (row, (a, b)) in compact.iter().zip(&full).enumerate() {
            assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "row {row}: compact KL {a} against the full-logit KL {b}");
        }
    }

    #[test]
    fn the_starting_library_of_a_qwen3_decoder_is_the_model_in_every_experiment() {
        let (native, layers, family, sequences) = tiny_qwen3("library_start_qwen3");
        let explanation = explanation(&native, &layers).unwrap();
        explanation.artifact.validate_coverage(&native).unwrap();
        assert_eq!(explanation.artifact.blocks.len(), 2 * (2 + 1), "two heads and one MLP per layer");
        // Per key-value group (one, read by both query heads) two rotary planes and four value rows;
        // per MLP function its gate, up direction and output.
        assert_eq!(explanation.groups.len(), 2 * ((2 + 4) + 16 * 3));
        assert!(explanation.layers.iter().all(|l| l.functions.iter().all(|f| f.len() == 3)));
        assert!(explanation.layers.iter().all(|l| l.heads[0] == l.heads[1]), "both query heads hold their group's planes and values");
        for l in 0..2 {
            assert_eq!(key_value(&explanation.artifact.program, l, 0), key_value(&explanation.artifact.program, l, 1), "one key and value map per key-value group");
        }
        let expected = native.execute(&family, false).unwrap().values[native.output].clone();
        let actual = explanation.artifact.execute(&family).unwrap().values[explanation.artifact.program.output].clone();
        let scale = expected.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let difference = expected.iter().zip(&actual).fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(difference <= 1e-12 * scale, "the starting library differs from the native model by {difference}");
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let cells: usize = explanation.groups.iter().flat_map(|g| &g.cells).map(|c| c.rows.len() * c.cols.len()).sum();
        assert_eq!(cells, posterior.mean.iter().map(Array2::len).sum::<usize>(), "the groups partition the parameters");
        let settings = settings();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings, "tiny", None).unwrap();
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 72.0, None, 0).unwrap();
        let evaluation = held_out(&mut scorer, &explanation, &posterior, &device_posterior, &sequences, &settings, 72, None).unwrap();
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
        let fit = fit(&Device::host(), &native, &explanation, train, held, &settings(), "tiny", None, None).unwrap();
        let report = &fit.report;
        assert_eq!(report.removals.last().unwrap().removed, 0, "the fit ends when no removal is accepted");
        assert!(report.removals.iter().all(|r| r.after_bits <= r.before_bits), "a removal never increases the objective");
        assert!(report.epochs.iter().all(|e| e.objective_bits.is_finite() && e.held_out.objective_bits_per_token.is_finite()));
        let fitted = posterior_mean(&explanation, &fit.posterior).unwrap();
        fitted.validate_coverage(&native).unwrap();
        // After training, both query heads still read one key and one value map, moved from M's.
        for l in 0..2 {
            let ((k, v), other) = (key_value(&fitted.program, l, 0), key_value(&fitted.program, l, 1));
            assert_eq!((k, v), other);
            assert_ne!(fitted.program.operators[k].matrix(), explanation.artifact.program.operators[k].matrix(), "training moves the shared key");
        }
    }

    /// The key and value operators head `h` of layer `l` reads.
    fn key_value(program: &OperatorProgram, l: usize, h: usize) -> (usize, usize) {
        let rule = program.rules.iter().find(|r| r.name == format!("library.l{l}.h{h}")).unwrap();
        let map = |node: usize| match &rule.nodes[node] {
            Node::Affine { terms, .. } => terms[0].1,
            other => panic!("{other:?}"),
        };
        (map(2), map(3))
    }

    #[test]
    fn every_parameter_block_names_the_native_parameter_it_replaces_and_survives_the_bytes() {
        for (native, layers, family, _) in [tiny("library_owners", "gelu_tanh"), tiny_qwen3("library_owners_qwen3")] {
            let explanation = explanation(&native, &layers).unwrap();
            let program = &explanation.artifact.program;
            let owners = &explanation.artifact.owners;
            for owner in owners {
                let op = &program.operators[index_of(program, &owner.operator).unwrap()];
                let source = native.operators.iter().find(|o| o.name == owner.native).expect("a native owner of M");
                assert!(owner.rows.end <= op.rows.width() && owner.cols.end <= op.cols.width());
                assert!(owner.native_rows.end <= source.rows.width() && owner.native_cols.end <= source.cols.width());
                assert_eq!((owner.rows.len(), owner.cols.len()), (owner.native_rows.len(), owner.native_cols.len()));
                // At the start, P's block is M's block.
                let (ours, theirs) = (op.matrix(), source.matrix());
                for (r, nr) in owner.rows.clone().zip(owner.native_rows.clone()) {
                    for (c, nc) in owner.cols.clone().zip(owner.native_cols.clone()) {
                        assert_eq!(ours[[r, c]], theirs[[nr, nc]], "{} against {}", owner.operator, owner.native);
                    }
                }
            }
            // Every trainable entry has an owner.
            for &op in &explanation.trainable {
                let name = &program.operators[op].name;
                let (rows, cols) = (program.operators[op].rows.width(), program.operators[op].cols.width());
                for r in 0..rows {
                    for c in 0..cols {
                        assert!(owners.iter().any(|o| &o.operator == name && o.rows.contains(&r) && o.cols.contains(&c)), "{name} ({r}, {c}) has no owner");
                    }
                }
            }
            // A key-value group's shared key is owned once per query head reading it.
            let key_sites: Vec<&str> = owners.iter().filter(|o| o.operator == "library.l0.kv0.k").map(|o| o.site.as_str()).collect();
            let heads_reading = program.rules.iter().filter(|r| r.name.starts_with("library.l0.h") && r.nodes.iter().any(|n| matches!(n, Node::Affine { terms, .. } if terms.iter().any(|t| program.operators[t.1].name == "library.l0.kv0.k")))).count();
            assert_eq!(key_sites.len(), heads_reading);
            let bytes = explanation.artifact.f32_literals().unwrap().to_bytes().unwrap();
            let decoded = Artifact::from_bytes(&bytes, &native.declarations).unwrap();
            assert_eq!(&decoded.owners, owners);
            let _ = family;
        }
    }

    #[test]
    fn the_reported_objective_pays_for_the_explanation_s_choices() {
        let (native, layers, _, sequences) = tiny("library_choice_bits", "gelu_tanh");
        let mut explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings, "tiny", None).unwrap();
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 72.0, None, 0).unwrap();
        let without = held_out(&mut scorer, &explanation, &posterior, &device_posterior, &sequences, &settings, 72, None).unwrap();
        explanation.fixed_nats = 1000_f64.ln();
        let with = held_out(&mut scorer, &explanation, &posterior, &device_posterior, &sequences, &settings, 72, None).unwrap();
        assert!((with.choice_bits - 1000_f64.log2()).abs() < 1e-12);
        let added = (with.objective_bits_per_token - without.objective_bits_per_token) * 72.0;
        assert!((added - 1000_f64.log2()).abs() < 1e-9, "F per token gains the choice's bits over the scored tokens: {added}");
    }

    #[test]
    fn a_tied_gate_is_differentiated_through_both_of_its_uses() {
        let (native, layers, _, sequences) = tiny("library_tie_gradient", "gelu_tanh");
        let start = explanation(&native, &layers).unwrap();
        let tie = crate::library_sharing::Tie { source: (0, 3), target: (1, 5), scale: 0.7 };
        let explanation = crate::library_sharing::tie(&start, &[tie]).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings, "tiny", None).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let (theta, _) = posterior.sample(noise_seed(settings.seed, 0, 0));
        let (_, gradients) = scorer.score(&batch, &experiments, &theta, "gradient", true).unwrap();
        let at = |name: &str| explanation.trainable.iter().position(|op| explanation.artifact.program.operators[*op].name == name).unwrap();
        let (gate, scale) = (at("library.l1.mlp.gate"), at("library.l0.mlp.tie1.f3.scale"));
        let mut bits = |theta: &[Array2<f64>]| -> f64 {
            scorer.score(&batch, &experiments, theta, "gradient", false).unwrap().0.iter().flatten().sum()
        };
        for (i, entry) in [(gate, (5, 0)), (gate, (5, 6)), (scale, (0, 0))] {
            let h = 1e-5;
            let (mut up, mut down) = (theta.clone(), theta.clone());
            up[i][entry] += h;
            down[i][entry] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!((gradients[i][entry] - central).abs() <= 1e-5 * (1.0 + central.abs()), "tied gradient {} against {central}", gradients[i][entry]);
        }
    }

    #[test]
    fn a_block_shared_across_layers_is_differentiated_through_both_of_its_uses() {
        let (native, layers, _, sequences) = tiny("library_block_gradient", "gelu_tanh");
        let start = explanation(&native, &layers).unwrap();
        let row = crate::library_sharing::tie_row(&start, "gate", (1, 7), crate::library_sharing::RowSource::Row { layer: 0, part: "gate", function: 4 }, 0.8).unwrap();
        let explanation = crate::library_sharing::tie_column(&row, (1, 7), (0, 4), -0.6).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings, "tiny", None).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let (theta, _) = posterior.sample(noise_seed(settings.seed, 0, 0));
        let (_, gradients) = scorer.score(&batch, &experiments, &theta, "gradient", true).unwrap();
        let at = |name: &str| explanation.trainable.iter().position(|op| explanation.artifact.program.operators[*op].name == name).unwrap();
        let (gate, out) = (at("library.l0.mlp.gate"), at("library.l0.mlp.out"));
        let (row_scale, column_scale) = (at("library.l1.mlp.f7.gate.from_l0_gate4.scale"), at("library.l1.mlp.f7.out.from_l0_4.scale"));
        let mut bits = |theta: &[Array2<f64>]| -> f64 { scorer.score(&batch, &experiments, theta, "gradient", false).unwrap().0.iter().flatten().sum() };
        for (i, entry) in [(gate, (4, 0)), (gate, (4, 5)), (out, (2, 4)), (out, (6, 4)), (row_scale, (0, 0)), (column_scale, (0, 0))] {
            let h = 1e-5;
            let (mut up, mut down) = (theta.clone(), theta.clone());
            up[i][entry] += h;
            down[i][entry] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!((gradients[i][entry] - central).abs() <= 1e-5 * (1.0 + central.abs()), "shared block gradient {} against {central}", gradients[i][entry]);
        }
    }

    #[test]
    fn a_shared_key_is_differentiated_through_every_query_head_reading_it() {
        let (native, layers, _, sequences) = tiny_qwen3("library_shared_key_gradient");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings, "tiny", None).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let (theta, _) = posterior.sample(noise_seed(settings.seed, 0, 0));
        let (_, gradients) = scorer.score(&batch, &experiments, &theta, "gradient", true).unwrap();
        let (k, _) = key_value(&explanation.artifact.program, 1, 0);
        let i = explanation.trainable.iter().position(|op| *op == k).unwrap();
        let mut bits = |theta: &[Array2<f64>]| -> f64 {
            scorer.score(&batch, &experiments, theta, "gradient", false).unwrap().0.iter().flatten().sum()
        };
        for at in [(0, 0), (1, 3), (3, 7)] {
            let h = 1e-5;
            let (mut up, mut down) = (theta.clone(), theta.clone());
            up[i][at] += h;
            down[i][at] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!((gradients[i][at] - central).abs() <= 1e-5 * (1.0 + central.abs()), "shared key gradient {} against {central}", gradients[i][at]);
        }
    }

    fn settings() -> Settings {
        Settings {
            batch_sequences: 2,
            rate: 0.1,
            beta1: 0.9,
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
            // `M`'s MLPs have no bias, so neither do the library's: a zero input still maps to zero.
            for l in 0..2 {
                assert!(operator_named(&explanation.artifact.program, &format!("library.l{l}.mlp.gate_bias")).is_none());
                let rule = explanation.artifact.program.rules.iter().find(|r| r.name == format!("library.l{l}.mlp")).unwrap();
                assert!(matches!(rule.nodes[1], Node::Affine { bias: None, .. }));
            }
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
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings, "tiny", None).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let evaluation = held_out(&mut scorer, &explanation, &posterior, &device_posterior, &sequences, &settings, 72, None).unwrap();
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
    fn removal_scores_the_fixed_collection() {
        // Removing every group changes neither the experiments nor their directions: a trial is
        // scored on the collection drawn over M's reads, with M's directions and M's targets.
        let (native, layers, _, sequences) = tiny("library_removal_evidence", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 2 * sequences.len() * 12).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings, "tiny", None).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let removed: Vec<usize> = (0..posterior.active.len()).collect();
        let mut trial = posterior.clone();
        trial.remove(&removed);
        // The reference, built apart from Scorer::score: M's variables and the library's start.
        let variables = interchange::library_reads(&explanation.artifact.program, layers.len()).unwrap();
        let start: Vec<Array2<f64>> = explanation.trainable.iter().map(|op| explanation.artifact.program.operators[*op].matrix()).collect();
        let (mut reference_bits, mut read_patches) = (0.0, 0);
        for (b, draw) in draws.iter().enumerate() {
            let batch = draw.batch(&sequences).unwrap();
            let experiments = scorer.experiments(draw, &sequences).unwrap();
            read_patches += experiments.iter().filter(|e| matches!(e.patch, Some(Patch::Read { .. }))).count();
            let (theta, _) = trial.sample(noise_seed(settings.seed, 0, b));
            let design = scorer.experiments.design_at(&variables, &experiments, &start).unwrap();
            let targets = scorer.experiments.targets(&batch, &experiments, &design).unwrap();
            scorer.experiments.load(&theta).unwrap();
            let reference = scorer.experiments.evaluate_resident(&batch, &experiments, &design, &targets, false).unwrap();
            reference_bits += reference.bits.iter().flatten().sum::<f64>();
        }
        assert!(read_patches > 0, "the collection must hold read patches of removed functions");
        let actual = expected_divergence(&mut scorer, &posterior, &draws, &sequences, &removed, &settings, None).unwrap();
        let reference = reference_bits * LN_2;
        assert!((actual - reference).abs() < 1e-10 * reference.abs().max(1.0), "removal must score the fixed collection");
    }

    #[test]
    fn dead_groups_are_the_planes_of_silent_heads_and_the_rest_of_silent_functions() {
        let (native, layers, _, _) = tiny("library_dead", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let mut active = vec![true; explanation.groups.len()];
        assert!(dead_groups(&explanation, &active).is_empty(), "nothing is dead at the start");
        let layer = &explanation.layers[0];
        let (planes, values) = layer.heads[0].clone();
        for g in &values {
            active[*g] = false;
        }
        let (gate, output) = (layer.functions[0][0], *layer.functions[0].last().unwrap());
        let (other_gate, other_output) = (layer.functions[1][0], *layer.functions[1].last().unwrap());
        active[output] = false;
        active[other_gate] = false;
        let dead = dead_groups(&explanation, &active);
        let mut expected: Vec<usize> = planes.clone();
        expected.extend([gate, other_output]);
        expected.sort_unstable();
        assert_eq!(dead, expected, "a silent head's planes, a function's gate without its output, its output without its gate");
        for g in &dead {
            active[*g] = false;
        }
        assert!(dead_groups(&explanation, &active).is_empty(), "one pass reaches the fixed point");
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
        let tokens = 72.0;
        let mut device_posterior = DevicePosterior::new(&device, &explanation, &posterior, tokens, None, 0).unwrap();
        // A data gradient per operator, weighted by `scale`, at the sample whose noise the device
        // regenerates from `key` (stream `i` for operator `i`).
        let (key, scale) = (99, 3.0);
        let gradients: Vec<Array2<f64>> = posterior.mean.iter().map(|m| m.mapv(|_| rng.random::<f64>() - 0.5)).collect();
        let uploaded: BTreeMap<usize, Tensor> = explanation.trainable.iter().zip(&gradients).map(|(op, g)| (*op, device.upload(g.view()).unwrap())).collect();
        let ivon = Ivon { rate: settings.rate, beta1: settings.beta1, beta2: 0.75 };
        device_posterior.step(&uploaded, scale, &ivon, key).unwrap();
        let mut stepped = posterior.clone();
        device_posterior.download(&mut stepped).unwrap();
        // The host reference: IVON's first step from momentum zero and the curvature at which the
        // posterior's standard deviations are IVON's.
        let variance: Vec<f64> = posterior.moments().iter().map(|m| m.second / m.count).collect();
        let mut reference = posterior.clone();
        for i in 0..reference.mean.len() {
            let cols = reference.mean[i].ncols();
            for ((r, c), mu) in reference.mean[i].indexed_iter_mut() {
                let delta = 1.0 / (tokens * variance[posterior.membership[i][[r, c]] as usize]);
                let sd = posterior.log_sd[i][[r, c]].exp();
                let e = f64::from(posterior_normal(key, i as u64, (r * cols + c) as u64));
                let g = scale * gradients[i][[r, c]];
                let h0 = (1.0 / (tokens * sd * sd) - delta).max(0.0);
                let d = g * e / sd - h0;
                let h = (h0 + (1.0 - ivon.beta2) * d + 0.5 * (1.0 - ivon.beta2).powi(2) * d * d / (h0 + delta)).max(0.0);
                let momentum = (1.0 - ivon.beta1) * g;
                *mu -= ivon.rate * (momentum / (1.0 - ivon.beta1) + delta * *mu) / (h + delta);
                reference.log_sd[i][[r, c]] = -0.5 * (tokens * (h + delta)).ln();
            }
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
        let variance = (-3.0_f64).exp();
        let expected = 0.5 * (size as f64).ln() + scale_bits(variance, posterior.initial[0]) * LN_2;
        assert!((posterior.costs()[0] - expected).abs() < 1e-12, "its variance costs ½ ln |G| and its scale");
    }

    #[test]
    fn checkpoint_array_decoding_is_tiled_and_preserves_logical_order_and_bits() {
        let bits: Vec<u64> = (0..18_003)
            .map(|i| match i % 4 {
                0 => (-0.0_f64).to_bits(),
                1 => f64::INFINITY.to_bits(),
                2 => 0x7ff8_0000_0000_0123,
                _ => (i as f64 / 7.0).to_bits(),
            })
            .collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut reader = std::io::Cursor::new(&bytes);
        let mut output = Array2::zeros((6001, 3)).reversed_axes();
        assert!(!output.is_standard_layout());
        read_checkpoint_array(&mut reader, &mut output).unwrap();
        assert_eq!(reader.position(), bytes.len() as u64);
        assert_eq!(output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
        assert!(read_checkpoint_array(&mut &bytes[..bytes.len() - 1], &mut output).is_err());
    }

    fn checkpoint_fixture() -> (Progress, Posterior, Vec<u8>) {
        let shapes = vec![(2, 3), (0, 2), (3, 1)];
        let posterior = Posterior {
            mean: shapes.iter().map(|&dim| Array2::from_elem(dim, -7.0)).collect(),
            log_sd: shapes.iter().map(|&dim| Array2::from_elem(dim, -8.0)).collect(),
            active: vec![true, true],
            membership: Vec::new(),
            spans: Vec::new(),
            initial: Vec::new(),
        };
        let progress = Progress {
            identity: Identity {
                export: "export".into(),
                tokens: "tokens".into(),
                program: "program".into(),
                groups: "groups".into(),
                sharing: "sharing".into(),
            },
            settings: settings(),
            tokens: 23,
            shapes,
            start: None,
            epoch: 4,
            step: 17,
            epochs: Vec::new(),
            removals: Vec::new(),
            previous: Some(vec![1.0, 2.0]),
            active: vec![false, true],
            done: false,
            seconds: 5.0,
            training_seconds: 3.0,
            evaluation_seconds: 2.0,
            full_seconds: 1.0,
            prior: None,
        };
        // The wire format: length-prefixed JSON, then `CHECKPOINT_ARRAYS` row-major f64 arrays per
        // operator. This fixture is independent of the streaming decoder and requires no GPU.
        let header = serde_json::to_vec(&progress).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        for (i, &(rows, cols)) in progress.shapes.iter().enumerate() {
            for field in 0..CHECKPOINT_ARRAYS {
                for cell in 0..rows * cols {
                    bytes.extend(((100 * i + 10 * field + cell) as f64 + 0.25).to_le_bytes());
                }
            }
        }
        (progress, posterior, bytes)
    }

    #[test]
    fn checkpoint_streaming_restores_the_legacy_payload_without_replacing_posterior_arrays() {
        let (expected, mut posterior, bytes) = checkpoint_fixture();
        let path = std::env::temp_dir().join(format!("library_checkpoint_stream_{}.bin", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let pointers: Vec<_> = posterior.mean.iter().chain(&posterior.log_sd).map(|array| array.as_ptr()).collect();
        let (progress, moments) = load_checkpoint(&path, &expected, &mut posterior).unwrap();
        assert_eq!(serde_json::to_value(&progress).unwrap(), serde_json::to_value(&expected).unwrap());
        assert_eq!(posterior.active, expected.active);
        assert_eq!(pointers, posterior.mean.iter().chain(&posterior.log_sd).map(|array| array.as_ptr()).collect::<Vec<_>>());
        for i in 0..progress.shapes.len() {
            for (field, array) in [&posterior.mean[i], &posterior.log_sd[i]].into_iter().chain(&moments[i]).enumerate() {
                for (cell, value) in array.iter().enumerate() {
                    assert_eq!(*value, (100 * i + 10 * field + cell) as f64 + 0.25);
                }
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_streaming_refuses_truncation_trailing_bytes_and_overflow_before_mutating() {
        let (expected, posterior, bytes) = checkpoint_fixture();
        let path = std::env::temp_dir().join(format!("library_checkpoint_bad_{}.bin", std::process::id()));
        let mut trailing = bytes.clone();
        trailing.push(1);
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        for broken in [
            Vec::new(),
            bytes[..7].to_vec(),
            u64::MAX.to_le_bytes().to_vec(),
            bytes[..8 + header_len - 1].to_vec(),
            bytes[..bytes.len() - 1].to_vec(),
            trailing,
        ] {
            std::fs::write(&path, broken).unwrap();
            let mut unchanged = posterior.clone();
            assert!(load_checkpoint(&path, &expected, &mut unchanged).is_err());
            assert_eq!(unchanged.mean, posterior.mean);
            assert_eq!(unchanged.log_sd, posterior.log_sd);
            assert_eq!(unchanged.active, posterior.active);
        }
        std::fs::write(&path, &bytes).unwrap();
        let mut other = expected.clone();
        other.identity.groups.push('x');
        let message = load_checkpoint(&path, &other, &mut posterior.clone()).unwrap_err();
        assert!(message.contains("groups"), "{message}");
        other = expected.clone();
        other.tokens += 1;
        assert!(load_checkpoint(&path, &other, &mut posterior.clone()).unwrap_err().contains("another fit"));
        other = expected.clone();
        other.settings.seed += 1;
        assert!(load_checkpoint(&path, &other, &mut posterior.clone()).unwrap_err().contains("another fit"));
        assert_eq!(checkpoint_payload_bytes(&[(2, 3), (0, 2), (3, 1)]), Some(9 * 32));
        assert_eq!(checkpoint_payload_bytes(&[(usize::MAX, usize::MAX)]), None);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_identity_reads_a_bounded_prefix_without_loading_the_payload() {
        let (expected, _, bytes) = checkpoint_fixture();
        let path = std::env::temp_dir().join(format!("library_checkpoint_identity_{}.bin", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        // A sparse payload makes the bytes read observable independently of posterior size.
        std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(8 * 1024 * 1024).unwrap();
        check_checkpoint(&path, &expected.identity).unwrap();
        let (progress, mut reader, _): (Progress, _, _) = checkpoint_header(&path).unwrap();
        assert_eq!(progress.identity, expected.identity);
        assert!(std::io::Seek::stream_position(reader.get_mut()).unwrap() <= 64 * 1024);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn the_fit_converges_removes_and_reports_its_posterior_mean() {
        let (native, layers, _, sequences) = tiny("library_fit", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let (train, held) = sequences.split_at(4);
        let checkpoint = std::env::temp_dir().join(format!("library_fit_{}.bin", std::process::id()));
        let fit = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&checkpoint), None).unwrap();
        // A finished fit's checkpoint resumes to the same posterior without another step.
        let resumed = super::fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&checkpoint), None).unwrap();
        // A checkpoint of another export, other sequences or another library is refused, and so
        // is a fit that would resume from it.
        let own = identity("tiny", &native, &explanation, train, held);
        check_checkpoint(&checkpoint, &own).unwrap();
        let mut groups = explanation.clone();
        groups.groups.swap(0, 1);
        let (mut swapped, mut shared) = (train.to_vec(), explanation.clone());
        swapped.swap(0, 1);
        shared.artifact.program.rules[0].name.push('x');
        // Head 1 of layer 0 reading head 0's key: the shared-parameter map changes.
        let mut tied = explanation.clone();
        let (key, _) = key_value(&tied.artifact.program, 0, 0);
        let rule = tied.artifact.program.rules.iter_mut().find(|r| r.name == "library.l0.h1").unwrap();
        rule.nodes[2] = Node::Affine { terms: vec![(0, key)], bias: None };
        for (other, field) in [
            (identity("another", &native, &explanation, train, held), "export"),
            (identity("tiny", &native, &explanation, &swapped, held), "tokens"),
            (identity("tiny", &native, &shared, train, held), "program"),
            (identity("tiny", &native, &groups, train, held), "groups"),
            (identity("tiny", &native, &tied, train, held), "sharing"),
        ] {
            let refusal = check_checkpoint(&checkpoint, &other).unwrap_err();
            assert!(refusal.contains(field), "{refusal}");
        }
        assert!(super::fit(&device, &native, &explanation, train, held, &settings, "another", Some(&checkpoint), None).is_err());
        std::fs::remove_file(&checkpoint).unwrap();
        std::fs::remove_file(checkpoint.with_extension("json")).unwrap();
        std::fs::remove_file(checkpoint.with_extension("artifact.bin")).unwrap();
        std::fs::remove_dir_all(checkpoint.with_extension("targets")).unwrap();
        assert_eq!(resumed.posterior.active, fit.posterior.active);
        assert_eq!(resumed.posterior.means(), fit.posterior.means());
        assert_eq!(resumed.report.epochs.len(), fit.report.epochs.len());
        let report = &fit.report;
        // Every clean experiment scores its 12 tokens, every patched one those from its position on.
        assert!(report.scored_tokens > 4 * 12 && report.scored_tokens <= 2 * 4 * 12, "{}", report.scored_tokens);
        assert_eq!(report.families.values().sum::<usize>(), 2 * 4, "two experiments per training base");
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

    #[test]
    fn every_candidate_is_asked_the_native_questions() {
        // A warm start from moved values, and a tied candidate built from it, are asked the native
        // questions: bitwise the same experiments, directions and targets as the start at M.
        let (native, layers, _, sequences) = tiny("library_protocol", "gelu_tanh");
        let start = explanation(&native, &layers).unwrap();
        let mut fitted = start.artifact.clone();
        for (k, &op) in start.trainable.iter().enumerate() {
            let old = Arc::clone(&fitted.program.operators[op]);
            let values = old.matrix().mapv(|v| v * (1.0 + 0.1 * (k as f64 + 7.0 * v).sin()));
            let precision = exact_precision(values.iter().copied()).unwrap();
            fitted.program.operators[op] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).unwrap());
        }
        let warm = crate::library_sharing::warm(&start, &fitted).unwrap();
        let tied = crate::library_sharing::tie(&warm, &[crate::library_sharing::Tie { source: (0, 0), target: (1, 1), scale: 0.5 }]).unwrap();
        let settings = settings();
        let device = Device::host();
        let mut scorers: Vec<Scorer> = [&start, &warm, &tied].iter().map(|e| Scorer::new(&device, &native, e, &settings, "tiny", None).unwrap()).collect();
        for (b, draw) in draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap().iter().enumerate() {
            let batch = draw.batch(&sequences).unwrap();
            let experiments: Vec<Vec<Experiment>> = scorers.iter().map(|s| s.experiments(draw, &sequences).unwrap()).collect();
            assert!(experiments.iter().all(|e| *e == experiments[0]), "batch {b}: the experiments");
            let made: Vec<(u64, Vec<(Array2<f64>, Vec<f64>)>)> = scorers
                .iter_mut()
                .map(|s| {
                    let (design, targets) = s.targets(&batch, &experiments[0], &format!("train_{b}")).unwrap();
                    (design.fingerprint(&device).unwrap(), targets.host(&device).unwrap())
                })
                .collect();
            assert!(made.iter().all(|m| *m == made[0]), "batch {b}: the directions and targets");
        }
    }

    #[test]
    fn the_held_out_sample_follows_the_training_weights() {
        // Held-out experiments are a sample of the training distribution, not an enumeration of
        // hybrid sizes: one clean and one patched experiment per base, each family's count within
        // five standard deviations of its stated weight (four blocks: P alone 1/2 + 1/2 * 1/4;
        // attention and MLP single reads 1/4 each; joint reads 1/2).
        let (native, layers, _, sequences) = tiny("library_held_out_sample", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let scorer = Scorer::new(&Device::host(), &native, &explanation, &settings, "tiny", None).unwrap();
        let held: Vec<Vec<u32>> = sequences.iter().cycle().take(2000).cloned().collect();
        let sampled = held_out_experiments(&scorer, &held, &settings).unwrap();
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for (draw, experiments) in &sampled {
            assert_eq!(experiments.len(), 2 * draw.bases.len());
            for (family, count) in interchange::census(experiments, scorer.protocol.variables()) {
                *counts.entry(family).or_default() += count;
            }
        }
        let n = held.len() as f64;
        let blocks = 2.0 * layers.len() as f64;
        for (family, p) in [("clean_alone", 0.5 + 0.5 / blocks), ("read_attention", 0.25), ("read_mlp", 0.25), ("read_joint", 0.5)] {
            let (mean, sd) = (n * p, (n * p * (1.0 - p)).sqrt());
            assert!((counts[family] as f64 - mean).abs() <= 5.0 * sd, "{family}: {} against {mean} ± {sd}", counts[family]);
        }
    }
}
