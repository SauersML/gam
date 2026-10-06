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
//! a random subset of the block's variables, its size uniform from one to all of them and then the
//! subset uniform; then a position). A patch replaces the patched functions' read values by their
//! values on the source, in `M` at `M`'s functions and in `P` at the call sites that replaced them
//! (`interchange::values`). The variables are `M`'s functions (`interchange::reads`), those `P`
//! removes included, one at a time and jointly: a function `P` no longer computes is patched in
//! `M` alone, which asks whether it matters. A base's source is another training sequence, drawn
//! uniformly. The questions do not move as `P` learns or loses functions, so `F`, the convergence
//! test and every removal comparison score the same experiments, and `M`'s targets for them are
//! made on the device whenever a batch is scored (`interchange::targets`).
//!
//! An explanation of some blocks only ([`scoped`]) is `M` everywhere else: its library holds the
//! trainable operators and prior groups of those blocks alone, so only they are charged, and every
//! experiment runs `P`'s blocks exactly there and `M`'s elsewhere (no other hybrid would differ from
//! `M`), with the patch drawn as above over all `2L` blocks.
//!
//! # The fit
//!
//! Each step draws one weight sample `θ = μ + σ ⊙ ε` (`ε` standard normal), runs a batch's
//! experiments, and takes one step of the improved variational online Newton method (IVON; Shen et
//! al., ICML 2024) on `F / N = E_q[ℓ] + KL(q ‖ p) / N`, `ℓ` the data term per scored token and `N`
//! the scored tokens of every training experiment. The data term's curvature is taken in the
//! Gauss–Newton approximation: per token the diagonal of `Σ_t J_tᵀ F_t J_t`, `F_t` the Fisher
//! matrix of `P_e`'s softmax at token `t`, which is positive semidefinite. The Hessian adds the
//! second derivatives of `P_e`'s logits weighted by `p_P − p_M`, so the two agree where `P_e`'s
//! predictions equal `M_e`'s. The estimate is the squared Gauss–Newton factor `ĥ = u ⊙ u / n`
//! (`interchange::Factor`): `u` is the gradient of `Σ_t log P_e(y_t)` over the batch's `n` scored
//! tokens, each label `y_t` drawn from `P_e`'s own prediction, from a second reverse pass through
//! the step's forward pass, and `E[ĥ]` is that diagonal. The reparameterization estimate
//! `g ε / σ` of the Hessian's diagonal carries every other weight's noise through the off-diagonal
//! terms (on vpd4l, eight draws showed no signal). `h` is the estimates' average over the last
//! epoch's batches (`β₂ = 1 − 1/B` for `B` training batches), starting from the Laplace start
//! (`laplace_start`): one pass over the collection at a weight sample of the unit-information
//! posterior (`σ² = v_G / N`) draws every batch's factor, `h = Σ_b u_b ⊙ u_b / N`, and
//! `σ² = 1 / (N h + 1 / v_G)`. From the unit-information curvature `1 / v_G` instead, `h` fell by
//! one e-fold per epoch toward the measured curvature, and `F` by a fixed 1.87e7 bits per epoch
//! for 10 epochs (vpd4l, `N = 2^16`; 17 epochs at `2^24`). Under the approximation `h ≥ 0`, and
//! the approximated
//! objective is stationary in `σ` at `σ = 1 / √(N (h + δ))`, `δ = 1 / (N v_G)` the group prior's
//! precision per token, so `σ² ≤ v_G`. That stationary point is implicit (`h` is an expectation
//! under `q`, and `v_G` depends on `σ`); setting `σ` from the running `h` and the current `v_G` at
//! every step is an online approximation to it, so `σ` has no step size; the mean takes the
//! preconditioned
//! step `α (ĝ + δ μ) / (h + δ)`, `ĝ` the gradient's momentum filtered by its own measured noise
//! (`Device::posterior_ivon`): a sampled gradient is mostly the other weights' noise carried
//! through the Hessian's off-diagonal terms, and with the momentum unfiltered the mean's steps
//! walked it away from `M` and raised `F` (vpd4l, `N = 2^16`: from the Laplace posterior at `M`,
//! `F` rose from 147 to 356 bits per scored token over the first epoch with the curvature held
//! fixed, and fell to 54 with the means held fixed). The posterior stays on the
//! device through an epoch (`device_posterior`): the sample is written into the explanation's
//! program, the gradient stays where the reverse pass left it, and the IVON step and the groups'
//! divergences run there; the host holds it between epochs, for the
//! held-out evaluation, the checkpoint and the removal step. An epoch visits every training batch
//! once, in a fixed order. The continuous fit stops descending by a statistical criterion, not a
//! proof of stationarity: when an epoch's mean improvement of the per-batch estimate of `F` over
//! the previous epoch, paired by batch (the same batches, experiments and noise seeds' structure),
//! is within its standard error. The criterion is on `F` itself rather than on the natural
//! gradient in `μ`: under IVON the means settle long before the curvature `h`, and so `σ`, has
//! finished its epoch-scale decay, which only `F` sees.
//!
//! A converged fit then removes groups (`library_removal`): first every group without effect on
//! any experiment (a rotary plane of a head with no value coordinate left, the gate of a function
//! whose output is removed), found exactly from the program's structure; then units of groups (a
//! group with those its removal silences) ranked by their second-order removal effect (below),
//! tested in segments of the ranked list by galloping and bisection, each segment ending on a unit
//! the objective rejects and the next starting after it.
//! Where a proposal deletes functions of an MLP, the MLP's surviving functions' outputs move by the
//! least-squares solution that takes over the deleted functions' output on `P`'s own states
//! (`library_compensation`), and the comparison scores the removal with those outputs. The
//! objective is estimated over the whole training set with one common weight sample per batch for
//! both sides of every comparison, on the fixed collection: acceptance never increases that
//! realization of the sampled `F`, whose expectation over the posterior it estimates without
//! bounding. Every proposal and its outcome go to a JSON-lines log next to the checkpoint. The fit
//! alternates converging and removing; it stops when a round accepts nothing, which says the search
//! found no removal, not that none exists.
//!
//! The removal search orders its proposals by each group's second-order removal effect: the rise of
//! the expected data term when the group's means and noise become exactly zero, less the description
//! it saves. Expanding the data term `D` about the posterior mean `μ`, with `g` its gradient there
//! and `H` its Gauss–Newton matrix (the Hessian of `Σ KL(M_e ‖ P_e)` where `P_e`'s predictions equal
//! `M_e`'s), the rise is `−g_G · μ_G + ½ μ_Gᵀ H μ_G − ½ Σ_{j∈G} H_jj σ_j²`. Both are measured on the
//! fixed collection at the posterior mean, one forward pass per training batch reversed twice
//! (`interchange::evaluate_labelled`): once for `g`, and once for a draw `u` of the gradient of
//! `Σ log P_e(y)` with every `y` drawn from `P_e` itself, whose `E[(u_G · μ_G)²] = μ_Gᵀ H μ_G` and
//! `E[u_j²] = H_jj` ([`Curvature`]). The form `μ_Gᵀ H μ_G` keeps the couplings between a group's
//! parameters (an output vector's direction against the downstream metric) that a diagonal
//! curvature drops. A deletion is a finite step and the fit is not shown to be stationary, so the
//! estimate orders proposals; the exact evaluation decides them.
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
    device_program::{gelu_tanh_constant, law_of},
    interchange::{self, Batch, Experiment, Interchange, Patch, ReadVariable, Targets},
    library_compensation::Compensation,
    library_removal::{self, Search},
    operator_program::{
        FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram, Provenance, Rule, SequenceLayout, SlotValues,
        exact_precision,
    },
    run_check::{LayerNodes, head_projection},
};
use gam_gpu::tensor::{Device, Op, Storage, Tensor};
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
    path::Path,
    sync::Arc,
    thread::JoinHandle,
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
    /// Per group, the reference variance `v⁰_G` its variance's scale is sent against (module
    /// note): the mean square of `M`'s values over its cells in the explanation made at `M`, and
    /// for a group a rewrite or share makes, a value fixed by `M` and the structure. It stays with
    /// the group when the explanation starts from moved values (a warm start), so the scale code
    /// is decodable from `M` and the explanation's structure.
    pub reference: Vec<f64>,
}

/// Per group of `groups`, the mean square of `program`'s values over its cells.
pub fn mean_squares(program: &OperatorProgram, groups: &[Group]) -> Vec<f64> {
    let mut matrices: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
    groups
        .iter()
        .map(|group| {
            let (mut count, mut sum) = (0.0, 0.0);
            for cell in &group.cells {
                let matrix = matrices.entry(cell.operator).or_insert_with(|| program.operators[cell.operator].matrix());
                for &row in &cell.rows {
                    for col in cell.cols.clone() {
                        sum += matrix[[row, col]] * matrix[[row, col]];
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 { sum / count } else { 0.0 }
        })
        .collect()
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
            role: part.to_string(),
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
            for (operator, source, role) in [(format!("{name}.q"), &q, "q"), (format!("{shared}.k"), &k, "k"), (format!("{shared}.v"), &v, "v")] {
                let (rows, cols) = (0..source.rows.width(), 0..source.cols.width());
                owners.push(Owner { operator, rows: rows.clone(), cols: cols.clone(), body: name.clone(), site: name.clone(), native: source.name.clone(), native_rows: rows, native_cols: cols, role: role.into(), ..Owner::default() });
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
    let reference = mean_squares(&artifact.program, &groups);
    Ok(Explanation { artifact, trainable, groups, layers: out, removed: Vec::new(), fixed_nats: 0.0, reference })
}

/// The prior groups of each of `explanation`'s `2L` blocks (block `2l` layer `l`'s attention,
/// `2l + 1` its MLP), from its layers' heads and functions.
fn block_groups(explanation: &Explanation) -> Vec<Vec<usize>> {
    explanation
        .layers
        .iter()
        .flat_map(|layer| {
            let attention = layer.heads.iter().flat_map(|(planes, values)| planes.iter().chain(values)).copied().collect();
            [attention, layer.functions.iter().flatten().copied().collect()]
        })
        .collect()
}

/// Which of `explanation`'s `2L` blocks its library explains: those holding a trainable operator.
fn scope(explanation: &Explanation) -> Vec<bool> {
    let trainable: std::collections::BTreeSet<usize> = explanation.trainable.iter().copied().collect();
    block_groups(explanation).iter().map(|groups| groups.iter().any(|g| explanation.groups[*g].cells.iter().any(|c| trainable.contains(&c.operator)))).collect()
}

/// `explanation` restricted to `blocks` (block `2l` layer `l`'s attention, `2l + 1` its MLP): `M`
/// everywhere else. Only those blocks' prior groups and their operators stay trainable and charged;
/// the other blocks' library operators keep `M`'s values and no experiment runs them (module note).
/// A group in two blocks (a share across them) is refused.
pub fn scoped(explanation: &Explanation, blocks: &[usize]) -> Result<Explanation, String> {
    let per_block = block_groups(explanation);
    if blocks.is_empty() || blocks.iter().any(|b| *b >= per_block.len()) {
        return Err(format!("blocks {blocks:?} of {}", per_block.len()));
    }
    let mut block_of = vec![None; explanation.groups.len()];
    for (b, groups) in per_block.iter().enumerate() {
        for g in groups {
            if block_of[*g].replace(b).is_some_and(|other| other != b) {
                return Err(format!("{}: a group in two blocks", explanation.groups[*g].name));
            }
        }
    }
    // Old group index to new, for the kept groups in order.
    let mut kept = vec![None; explanation.groups.len()];
    let mut count = 0;
    for (g, block) in block_of.iter().enumerate() {
        if block.is_some_and(|b| blocks.contains(&b)) {
            kept[g] = Some(count);
            count += 1;
        }
    }
    let renumber = |ids: &[usize]| -> Vec<usize> { ids.iter().filter_map(|g| kept[*g]).collect() };
    let groups: Vec<Group> = explanation.groups.iter().zip(&kept).filter(|(_, k)| k.is_some()).map(|(g, _)| g.clone()).collect();
    let mut trainable: Vec<usize> = groups.iter().flat_map(|g| g.cells.iter().map(|c| c.operator)).collect();
    trainable.sort_unstable();
    trainable.dedup();
    let layers = explanation
        .layers
        .iter()
        .enumerate()
        .map(|(l, layer)| Layer {
            sites: layer.sites.clone(),
            heads: if blocks.contains(&(2 * l)) { layer.heads.iter().map(|(planes, values)| (renumber(planes), renumber(values))).collect() } else { Vec::new() },
            functions: if blocks.contains(&(2 * l + 1)) { layer.functions.iter().map(|f| renumber(f)).collect() } else { Vec::new() },
        })
        .collect();
    let reference = explanation.reference.iter().zip(&kept).filter(|(_, k)| k.is_some()).map(|(r, _)| *r).collect();
    Ok(Explanation { trainable, groups, layers, removed: renumber(&explanation.removed), reference, ..explanation.clone() })
}

// ------------------------------------------------------------------------------------- posterior

/// Per prior group, sums over draws of the sampled-label gradient `u` (`interchange::sampled_label`)
/// at the posterior mean, each draw weighted by the data term's weight on its tokens: `Σ (u_G · μ_G)²`,
/// the Gauss–Newton quadratic form of the group's means, and `Σ Σ_{j∈G} u_j² σ_j²`, that of its
/// noise; and over the same batches `Σ g_G · μ_G`, the data term's gradient `g` at the mean along
/// the group's means (module note).
#[derive(Clone, Debug)]
pub struct Curvature {
    pub(crate) quadratic: Vec<f64>,
    pub(crate) noise: Vec<f64>,
    pub(crate) slope: Vec<f64>,
}

impl Curvature {
    /// No draws yet, for `groups` prior groups.
    #[must_use]
    pub fn new(groups: usize) -> Self {
        Self { quadratic: vec![0.0; groups], noise: vec![0.0; groups], slope: vec![0.0; groups] }
    }
}

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
    /// Per group, the reference variance `v⁰_G` against which its variance's scale is sent
    /// (`Explanation::reference`).
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
        if explanation.reference.len() != explanation.groups.len() {
            return Err("one reference variance per prior group required".into());
        }
        let initial = explanation.reference.clone();
        let mut posterior = Self { mean, log_sd, active: vec![true; explanation.groups.len()], membership, spans, initial };
        posterior.remove(&explanation.removed);
        Ok(posterior)
    }

    /// The posterior over parameter arrays `mean` whose entries' prior groups are `membership`
    /// (one id array per mean array, ids below `groups`, every group holding an entry), every group
    /// in the explanation, at [`Posterior::new`]'s start: each group's `v⁰_G` the mean square of its
    /// starting means, and its standard deviations `√(v⁰_G / N)` for `tokens` training tokens `N`.
    /// A parameterization that is not a library explanation (the toy accounts of
    /// `mpd_toy_gate_2951`) is priced by this posterior's code length.
    pub fn from_parts(mean: Vec<Array2<f64>>, membership: Vec<Array2<u32>>, groups: usize, tokens: usize) -> Result<Self, String> {
        if membership.len() != mean.len() || membership.iter().zip(&mean).any(|(ids, m)| ids.dim() != m.dim()) {
            return Err("one group array per mean array, of its shape, required".into());
        }
        if tokens == 0 {
            return Err("no training tokens".into());
        }
        let mut squares = vec![(0.0, 0.0); groups];
        let mut spans: Vec<Range<usize>> = vec![usize::MAX..0; mean.len()];
        for ((values, ids), span) in mean.iter().zip(&membership).zip(&mut spans) {
            for (value, id) in values.iter().zip(ids.iter()) {
                let g = *id as usize;
                let entry = squares.get_mut(g).ok_or_else(|| format!("group {g} of {groups}"))?;
                entry.0 += 1.0;
                entry.1 += value * value;
                *span = span.start.min(g)..span.end.max(g + 1);
            }
        }
        if let Some(g) = squares.iter().position(|(count, sum)| !(*count > 0.0 && *sum > 0.0 && sum.is_finite())) {
            return Err(format!("group {g} holds no entry or starts at zero, and has no scale"));
        }
        let log_sd = membership
            .iter()
            .map(|ids| {
                ids.mapv(|id| {
                    let (count, sum) = squares[id as usize];
                    0.5 * (sum / count / tokens as f64).ln()
                })
            })
            .collect();
        let initial = squares.iter().map(|(count, sum)| sum / count).collect();
        Ok(Self { mean, log_sd, active: vec![true; groups], membership, spans, initial })
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

    /// Adds one draw `u` of the sampled-label gradient at the posterior mean (per trainable operator,
    /// in `Explanation::trainable` order; `interchange::sampled_label`) to `curvature`, its terms
    /// weighted by `weight`, the data term's weight on the draw's tokens.
    pub fn add_curvature(&self, u: &[Array2<f64>], weight: f64, curvature: &mut Curvature) -> Result<(), String> {
        if u.len() != self.mean.len() || u.iter().zip(&self.mean).any(|(a, b)| a.dim() != b.dim()) {
            return Err("one gradient per trainable operator, of its shape, required".into());
        }
        if curvature.quadratic.len() != self.active.len() {
            return Err("a curvature of another explanation".into());
        }
        // Per group, `u_G · μ_G` and `Σ u_j² σ_j²` of this draw.
        let partial: Vec<(Range<usize>, Vec<(f64, f64)>)> = (0..self.mean.len())
            .into_par_iter()
            .map(|i| {
                let span = self.spans[i].clone();
                let mut local = vec![(0.0, 0.0); span.len()];
                for (((g, mu), s), group) in u[i].iter().zip(self.mean[i].iter()).zip(self.log_sd[i].iter()).zip(self.membership[i].iter()) {
                    let at = *group as usize;
                    if self.active[at] {
                        let entry = &mut local[at - span.start];
                        entry.0 += g * mu;
                        entry.1 += g * g * (2.0 * s).exp();
                    }
                }
                (span, local)
            })
            .collect();
        let mut draw = vec![(0.0, 0.0); self.active.len()];
        for (span, local) in partial {
            for (g, (dot, noise)) in span.zip(local) {
                draw[g].0 += dot;
                draw[g].1 += noise;
            }
        }
        for (g, (dot, noise)) in draw.into_iter().enumerate() {
            curvature.quadratic[g] += weight * dot * dot;
            curvature.noise[g] += weight * noise;
        }
        Ok(())
    }

    /// Adds the data term's gradient `g` in nats at the posterior mean on one batch (per trainable
    /// operator, in `Explanation::trainable` order), weighted by `weight`, to `curvature`'s
    /// `Σ g_G · μ_G`.
    pub fn add_slope(&self, g: &[Array2<f64>], weight: f64, curvature: &mut Curvature) -> Result<(), String> {
        if g.len() != self.mean.len() || g.iter().zip(&self.mean).any(|(a, b)| a.dim() != b.dim()) {
            return Err("one gradient per trainable operator, of its shape, required".into());
        }
        if curvature.slope.len() != self.active.len() {
            return Err("a curvature of another explanation".into());
        }
        let partial: Vec<(Range<usize>, Vec<f64>)> = (0..self.mean.len())
            .into_par_iter()
            .map(|i| {
                let span = self.spans[i].clone();
                let mut local = vec![0.0; span.len()];
                for ((gradient, mu), group) in g[i].iter().zip(self.mean[i].iter()).zip(self.membership[i].iter()) {
                    let at = *group as usize;
                    if self.active[at] {
                        local[at - span.start] += gradient * mu;
                    }
                }
                (span, local)
            })
            .collect();
        for (span, local) in partial {
            for (at, dot) in span.zip(local) {
                curvature.slope[at] += weight * dot;
            }
        }
        Ok(())
    }

    /// Per group, the second-order rise of the expected data term in nats when it alone is removed,
    /// `−g_G · μ_G + ½ μ_Gᵀ H μ_G − ½ Σ_{j∈G} H_jj σ_j²` (module note), from `curvature` (zero for a
    /// removed group): a proposal score, not a bound on the finite change.
    pub fn removal_data(&self, curvature: &Curvature) -> Vec<f64> {
        (0..self.active.len())
            .map(|g| if self.active[g] { -curvature.slope[g] + 0.5 * curvature.quadratic[g] - 0.5 * curvature.noise[g] } else { 0.0 })
            .collect()
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

    /// The posterior with each mean rounded to its precision: to the nearest multiple of
    /// `2^⌊log2 σ_j⌋`, the coarsest power of two not above its standard deviation (a removed entry
    /// stays zero).
    #[must_use]
    pub fn rounded(&self) -> Self {
        let mut out = self.clone();
        for (mean, log_sd) in out.mean.iter_mut().zip(&self.log_sd) {
            ndarray::Zip::from(mean).and(log_sd).for_each(|mu, s| {
                if s.is_finite() {
                    let step = (s / LN_2).floor().exp2();
                    *mu = (*mu / step).round() * step;
                }
            });
        }
        out
    }

    /// `Σ_G KL(q_G ‖ p_G)`, the active groups' variances and which groups are active, in nats.
    pub fn description(&self) -> f64 {
        self.costs().iter().sum::<f64>() + self.subset_nats()
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
    /// The held-out data term per scored token on the same experiments at the posterior mean, and
    /// at the mean rounded to its posterior precision ([`Posterior::rounded`]): the two
    /// candidates for the reported artifact, judged against the samples' `data_bits_per_token`.
    #[serde(default)]
    pub mean_bits_per_token: f64,
    #[serde(default)]
    pub rounded_bits_per_token: f64,
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
    /// The groups removed as without effect on any experiment, which `removed` counts.
    #[serde(default)]
    pub dead: usize,
    /// `F` before and after, in bits.
    pub before_bits: f64,
    pub after_bits: f64,
    /// Every evaluated proposal `(groups, F − F_before)` in bits, `F_before` the state it was
    /// proposed on, and each unit a segment of the search ended on `(its first group, its own
    /// change of F)`.
    pub evaluations: Vec<(usize, f64)>,
    #[serde(default)]
    pub singles: Vec<(usize, f64)>,
    /// The units the round ended without testing alone: those predicted not to lower `F` after the
    /// first of them, tested only jointly (`library_removal`'s module note).
    #[serde(default)]
    pub untested: usize,
}

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
    /// The reported artifact: the posterior mean, or the mean rounded to its posterior precision,
    /// whichever has the lower held-out data term at the end ([`HeldOut::mean_bits_per_token`]).
    pub representative: Representative,
    /// The literals every evaluation ran with: f32 where the device held f32 (every operator
    /// rounded to f32 as it was uploaded), float64 on the host. The saved artifacts hold the same
    /// ([`Fit::artifact`]), so the reported fidelity is the saved artifact's.
    pub literals: Literals,
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

/// The literals a fit's evaluations ran with, and its saved artifacts hold (`Report::literals`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Literals {
    F32,
    F64,
}

impl Literals {
    /// The literals of a fit on `device`.
    #[must_use]
    pub fn of(device: &Device) -> Self {
        if device.float64() { Self::F64 } else { Self::F32 }
    }

    /// `artifact` with these literals: rounded to f32, or as it is.
    pub fn apply(self, artifact: Artifact) -> Result<Artifact, String> {
        match self {
            Self::F32 => artifact.f32_literals(),
            Self::F64 => Ok(artifact),
        }
    }
}

/// Which point of the posterior the reported artifact holds (`Report::representative`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Representative {
    Mean,
    Rounded,
}

impl Fit {
    /// The posterior whose means the reported artifact holds ([`posterior_mean`]).
    #[must_use]
    pub fn representative(&self) -> Posterior {
        match self.report.representative {
            Representative::Mean => self.posterior.clone(),
            Representative::Rounded => self.posterior.rounded(),
        }
    }

    /// The reported artifact of `explanation`: the representative's means with the literals the
    /// fit's evaluations ran with (`Report::literals`), so it is the artifact the held-out
    /// evaluation scored.
    pub fn artifact(&self, explanation: &Explanation) -> Result<Artifact, String> {
        self.report.literals.apply(posterior_mean(explanation, &self.representative())?)
    }
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
/// One of `batches` batches drawn uniformly from a collection of `tokens` scored tokens: the factor
/// turning the batch's data term into an unbiased estimate of the collection's (`B`), and the one
/// turning its gradient and squared Gauss–Newton factor into the collection's per token (`B / N`),
/// whatever the batch's share of the tokens. An epoch's mean of the estimates is the collection's
/// data term exactly.
fn batch_weights(batches: usize, tokens: usize) -> (f64, f64) {
    let b = batches as f64;
    (b, b / tokens as f64)
}

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
    /// The blocks the explanation explains ([`scope`]), when not all of them ([`scoped`]): every
    /// experiment runs `P` exactly there.
    scope: Option<Vec<bool>>,
}

impl Scorer {
    /// The experiments of `explanation` against `M`, the split native program `native`: the read
    /// variables patched are `M`'s functions (`interchange::reads`), each model patching its own
    /// value of each, so every explanation of the model is asked the same questions.
    fn new(device: &Device, native: &OperatorProgram, explanation: &Explanation, settings: &Settings) -> Result<Self, String> {
        let sites: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::reads(native, &sites)?;
        let experiments =
            Interchange::new(device, native, &sites, &explanation.artifact, &explanation.trainable, reads, settings.numeric_bytes, settings.head_tile_rows)?;
        let (flat, _, _) = interchange::sites(&explanation.artifact, &sites)?;
        let mlps = (0..sites.len()).map(|l| Mlp::of(&flat, l)).collect::<Result<_, _>>()?;
        let position = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let scope = Some(scope(explanation)).filter(|blocks| !blocks.iter().all(|b| *b));
        Ok(Self { experiments, mlps, position, scope })
    }

    fn layers(&self) -> usize {
        self.mlps.len()
    }

    fn at(&self, operator: usize) -> Result<usize, String> {
        self.position.get(&operator).copied().ok_or_else(|| format!("operator {operator} is not trainable"))
    }

    /// The batch's experiments from `draw`: the fixed collection's for that batch.
    fn experiments(&self, draw: &Draw, sequences: &[Vec<u32>]) -> Result<Vec<Experiment>, String> {
        let mut experiments = draw.experiments(sequences, self.experiments.variables(), 2 * self.layers())?;
        if let Some(scope) = &self.scope {
            for e in &mut experiments {
                e.explained.clone_from(scope);
            }
        }
        Ok(experiments)
    }

    /// The explanation on `experiments` (on `batch`) with the posterior on the device: at its
    /// weight sample of `sample` ([`DevicePosterior::sample_into`]), or at its mean when none, and
    /// with `gradient` the gradient of the sum per trainable operator and a draw of the
    /// Gauss–Newton factor (`interchange::Factor`, its labels drawn from the seed `sample`), left
    /// on the device; `M`'s targets made here.
    fn score_device(
        &mut self,
        posterior: &DevicePosterior,
        batch: &Batch,
        experiments: &[Experiment],
        sample: Option<u64>,
        gradient: bool,
    ) -> Result<(Vec<Vec<f64>>, BTreeMap<usize, Tensor>, Option<interchange::Factor>), String> {
        let targets = self.experiments.targets(batch, experiments)?;
        self.evaluate_device(posterior, (batch, experiments), sample, &targets, gradient)
    }

    /// [`Scorer::score_device`] against `M`'s `targets` already made for these experiments.
    fn evaluate_device(
        &mut self,
        posterior: &DevicePosterior,
        (batch, experiments): (&Batch, &[Experiment]),
        sample: Option<u64>,
        targets: &Targets,
        gradient: bool,
    ) -> Result<(Vec<Vec<f64>>, BTreeMap<usize, Tensor>, Option<interchange::Factor>), String> {
        match sample {
            Some(seed) => posterior.sample_into(self.experiments.explanation_mut(), seed)?,
            None => posterior.mean_into(self.experiments.explanation_mut())?,
        }
        let labels = match (gradient, sample) {
            (true, Some(seed)) => Some(uniforms(seed, batch, experiments)),
            _ => None,
        };
        let evaluation = self.experiments.evaluate_labelled(batch, experiments, Some(targets), gradient, labels.as_deref())?;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err("nonfinite explanation divergence".into());
        }
        Ok((evaluation.bits, evaluation.gradient, evaluation.factor))
    }
}

/// One uniform per scored row of `experiments` on `batch` (rows in experiment order) from `seed`:
/// the draws of the Gauss–Newton factor's labels (`interchange::evaluate_labelled`).
fn uniforms(seed: u64, batch: &Batch, experiments: &[Experiment]) -> Vec<f64> {
    let rows: usize = experiments.iter().map(|e| batch.length() - e.position).sum();
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows).map(|_| rng.random::<f64>()).collect()
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
/// distribution, one clean and one patched experiment per base. They are drawn from their own seed,
/// the SplitMix64 output after the fit's (`gam_linalg::utils::splitmix64_hash`): drawn from the
/// fit's seed itself, held-out batch `b` repeated training batch `b`'s interventions exactly, so
/// held-out scores tested unseen bases under seen interventions only.
fn held_out_experiments(scorer: &Scorer, sequences: &[Vec<u32>], settings: &Settings) -> Result<Vec<(Draw, Vec<Experiment>)>, String> {
    draws(sequences.len(), settings.batch_sequences, gam_linalg::utils::splitmix64_hash(settings.seed))?
        .into_iter()
        .map(|draw| {
            let experiments = scorer.experiments(&draw, sequences)?;
            Ok((draw, experiments))
        })
        .collect()
}

/// The held-out evaluation of `posterior` (held on the host, and as `device_posterior` on the
/// device) on `sequences` (module note); `tokens` is `N`.
fn held_out(
    scorer: &mut Scorer,
    explanation: &Explanation,
    (posterior, device_posterior): (&Posterior, &DevicePosterior),
    sequences: &[Vec<u32>],
    settings: &Settings,
    tokens: usize,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<HeldOut, String> {
    let blocks = 2 * scorer.layers();
    let (mut clean, mut patched) = (vec![Mean::default(); blocks], vec![Mean::default(); blocks]);
    let (mut read, mut joint, mut sampled) = (Mean::default(), Mean::default(), Mean::default());
    let (mut at_mean, mut at_rounded) = (Mean::default(), Mean::default());
    let size = |e: &Experiment| e.explained.iter().filter(|x| **x).count();
    // Each batch's targets from `M` are made once; the mean, the sample and the rounded posterior
    // (rounded on the device) are scored against them.
    let mut made = Vec::new();
    for (b, (draw, experiments)) in held_out_experiments(scorer, sequences, settings)?.into_iter().enumerate() {
        let batch = draw.batch(sequences)?;
        let targets = scorer.experiments.targets(&batch, &experiments)?;
        let (bits, _, _) = scorer.evaluate_device(device_posterior, (&batch, &experiments), None, &targets, false)?;
        bits.iter().for_each(|b| at_mean.add(b));
        for (e, bits) in experiments.iter().zip(&bits) {
            match &e.patch {
                None => clean[size(e) - 1].add(bits),
                Some(patch) => {
                    patched[size(e) - 1].add(bits);
                    match patch {
                        Patch::Read { .. } => read.add(bits),
                        Patch::Reads { .. } => joint.add(bits),
                    }
                }
            }
        }
        let (bits, _, _) = scorer.evaluate_device(device_posterior, (&batch, &experiments), Some(noise_seed(settings.seed, 0, b)), &targets, false)?;
        bits.iter().for_each(|b| sampled.add(b));
        made.push((batch, experiments, targets));
    }
    device_posterior.rounded_into(scorer.experiments.explanation_mut())?;
    for (batch, experiments, targets) in &made {
        let evaluation = scorer.experiments.evaluate_resident(batch, experiments, targets, false)?;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err("nonfinite explanation divergence".into());
        }
        evaluation.bits.iter().for_each(|b| at_rounded.add(b));
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
        mean_bits_per_token: at_mean.mean().ok_or("no held-out tokens")?,
        rounded_bits_per_token: at_rounded.mean().ok_or("no held-out tokens")?,
        divergence_bits: divergence / LN_2,
        variance_bits: (gaussian - divergence) / LN_2,
        choice_bits: explanation.fixed_nats / LN_2,
        prior_bits: prior_nats / LN_2,
        clean: clean.iter().map(Mean::mean).collect(),
        patched: patched.iter().map(Mean::mean).collect(),
        read_patch: read.mean(),
        joint_patch: joint.mean(),
        layers: activity(scorer, explanation, (posterior, device_posterior), sequences, settings)?,
    })
}

/// Per layer, the survivors of `posterior` and its functions' activity on `sequences` at the
/// posterior mean (`LayerCount`), counted on the device (`Device::resolved_counts`).
fn activity(scorer: &mut Scorer, explanation: &Explanation, (posterior, device_posterior): (&Posterior, &DevicePosterior), sequences: &[Vec<u32>], settings: &Settings) -> Result<Vec<LayerCount>, String> {
    device_posterior.mean_into(scorer.experiments.explanation_mut())?;
    let variance = |op: usize| -> Result<Array2<f64>, String> { Ok(posterior.log_sd[scorer.at(op)?].mapv(|s| (2.0 * s).exp())) };
    let (_, p) = scorer.experiments.models();
    let (program, d) = (p.program, p.program.device());
    // The counts are summed in float64 where the device holds it.
    let wide = match d.with_storage(Storage::F64) {
        Ok(wide) => wide,
        Err(_) => d.clone(),
    };
    let active = |g: &usize| posterior.active[*g];
    let mut out = Vec::with_capacity(scorer.layers());
    // Per map its weights' posterior variances and its bias's (zero without one), on the device.
    let variances = |map: &Map| -> Result<(Tensor, Tensor), String> {
        let weights = d.upload(variance(map.operator)?.view()).map_err(error)?;
        let bias = match map.bias {
            Some(b) => variance(b)?.column(0).to_vec(),
            None => vec![0.0; program.widths()[map.node]],
        };
        Ok((weights, d.upload_vec(1, bias.len(), bias).map_err(error)?))
    };
    // Per layer its surviving functions' flags, its law per function, and its maps' variances.
    let mut gates = Vec::with_capacity(scorer.layers());
    for (layer, mlp) in explanation.layers.iter().zip(&scorer.mlps) {
        let surviving: Vec<u32> = layer.functions.iter().map(|groups| u32::from(groups.iter().all(active))).collect();
        out.push(LayerCount {
            heads: layer.heads.iter().filter(|(_, values)| values.iter().any(active)).count(),
            planes: layer.heads.iter().map(|(planes, _)| planes.iter().filter(|g| active(g)).count()).sum(),
            values: layer.heads.iter().map(|(_, values)| values.iter().filter(|g| active(g)).count()).sum(),
            functions: surviving.iter().filter(|s| **s == 1).count(),
            nonzero_per_token: 0.0,
            resolved_per_token: 0.0,
        });
        // A layer whose MLP the explanation leaves to `M` ([`scoped`]) has no functions to count.
        if layer.functions.is_empty() {
            gates.push(None);
            continue;
        }
        let codes = d.upload_indices(&vec![law_of(mlp.law).code(); surviving.len()]).map_err(error)?;
        let alive = d.upload_indices(&surviving).map_err(error)?;
        gates.push(Some((alive, codes, variances(&mlp.gate)?, mlp.up.as_ref().map(variances).transpose()?)));
    }
    let mut rows = 0usize;
    for chunk in sequences.chunks(settings.batch_sequences) {
        let family = sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let trace = program.forward(&family)?;
        for ((count, mlp), (alive, codes, gate, up)) in out.iter_mut().zip(&scorer.mlps).zip(&gates).filter_map(|(pair, gates)| gates.as_ref().map(|g| (pair, g))) {
            let x = trace.value(mlp.input)?;
            let mut squares = d.empty(x.rows(), x.cols()).map_err(error)?;
            d.hadamard(&mut squares, x, x, false).map_err(error)?;
            // `s²` of each function's value at every token: `Σ_j σ²_j x_j²` plus the bias's `σ²`.
            let noise = |(weights, bias): &(Tensor, Tensor)| -> Result<Tensor, String> {
                let mut s2 = d.empty(x.rows(), weights.rows()).map_err(error)?;
                d.gemm(&mut s2, 1.0, &squares, Op::N, weights, Op::T, 0.0, program.arithmetic()).map_err(error)?;
                d.add_row(&mut s2, 1.0, bias).map_err(error)?;
                Ok(s2)
            };
            let (z, a) = (trace.value(mlp.gate.node)?, trace.value(mlp.activation)?);
            let sz2 = noise(gate)?;
            let mut sums = wide.zeros(1, 3).map_err(error)?;
            match (&mlp.up, up) {
                (Some(map), Some(variances)) => {
                    // A gated law: the value φ(z) y, its slope φ'(z) y against the gate's noise and
                    // φ(z) against the up direction's.
                    let y = trace.value(map.node)?;
                    let mut value = d.empty(a.rows(), a.cols()).map_err(error)?;
                    d.hadamard(&mut value, a, y, false).map_err(error)?;
                    let slope = d.law_slopes(y, z, codes, gelu_tanh_constant()).map_err(error)?;
                    d.resolved_counts((&value, &slope, a), (&sz2, Some(&noise(variances)?)), alive, &mut sums).map_err(error)?;
                }
                _ => {
                    let ones = d.broadcast_rows(&d.upload_vec(1, z.cols(), vec![1.0; z.cols()]).map_err(error)?, z.rows()).map_err(error)?;
                    let slope = d.law_slopes(&ones, z, codes, gelu_tanh_constant()).map_err(error)?;
                    d.resolved_counts((a, &slope, a), (&sz2, None), alive, &mut sums).map_err(error)?;
                }
            }
            let sums = wide.download(&sums).map_err(error)?;
            count.nonzero_per_token += sums[[0, 1]];
            count.resolved_per_token += sums[[0, 2]];
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
    /// The precision of the payload's `μ`, `ln σ`, momentum, curvature and gradient second-moment
    /// arrays: the storage the
    /// device posterior holds each in; none for a checkpoint written in float64 throughout.
    #[serde(default)]
    precision: Option<[Precision; CHECKPOINT_ARRAYS]>,
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
/// gradient's momentum, the curvature estimate and the gradient's second moment), each
/// little-endian in its [`Precision`].
const CHECKPOINT_ARRAYS: usize = 5;

/// The precision a checkpoint array is written in: the storage the device holds it in, so that
/// writing and reading it is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Precision {
    F64,
    F32,
    /// bfloat16: the 16 high bits of the value's f32.
    Bf16,
}

impl Precision {
    fn of(storage: Storage) -> Self {
        match storage {
            Storage::F64 => Self::F64,
            Storage::F32 => Self::F32,
            Storage::Bf16 => Self::Bf16,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }

    /// Each array kind's precision, float64 for a checkpoint that names none.
    fn of_payload(precision: Option<[Self; CHECKPOINT_ARRAYS]>) -> [Self; CHECKPOINT_ARRAYS] {
        precision.unwrap_or([Self::F64; CHECKPOINT_ARRAYS])
    }
}

/// The payload bytes of a checkpoint of operators of `shapes` with arrays in `precision`.
fn checkpoint_payload_bytes(shapes: &[(usize, usize)], precision: [Precision; CHECKPOINT_ARRAYS]) -> Option<u64> {
    let per_cell = precision.iter().map(|p| p.bytes() as u64).sum::<u64>();
    shapes.iter().try_fold(0_u64, |total, &(rows, cols)| {
        let cells = u64::try_from(rows).ok()?.checked_mul(u64::try_from(cols).ok()?)?;
        total.checked_add(cells.checked_mul(per_cell)?)
    })
}

/// Decode in fixed-size byte tiles directly into the destination, including nonstandard array
/// layouts. The wire order remains ndarray's logical iteration order used by the writer; every
/// precision widens exactly.
fn read_checkpoint_array(reader: &mut impl Read, array: &mut Array2<f64>, precision: Precision) -> Result<(), String> {
    let width = precision.bytes();
    let mut bytes = [0_u8; 64 * 1024];
    let mut remaining = array.len();
    let mut values = array.iter_mut();
    while remaining != 0 {
        let count = remaining.min(bytes.len() / width);
        reader.read_exact(&mut bytes[..count * width]).map_err(error)?;
        for (value, encoded) in values.by_ref().take(count).zip(bytes[..count * width].chunks_exact(width)) {
            *value = match precision {
                Precision::F64 => f64::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6], encoded[7]]),
                Precision::F32 => f64::from(f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]])),
                Precision::Bf16 => f64::from(f32::from_bits(u32::from(u16::from_le_bytes([encoded[0], encoded[1]])) << 16)),
            };
        }
        remaining -= count;
    }
    Ok(())
}

/// `array`'s values appended to `out` in `precision`, in ndarray's logical order; a value the
/// precision does not hold exactly (one the device could not have held) is refused.
fn write_checkpoint_array(out: &mut Vec<u8>, array: &Array2<f64>, precision: Precision) -> Result<(), String> {
    out.reserve(array.len() * precision.bytes());
    for &value in array {
        let narrow = value as f32;
        let exact = match precision {
            Precision::F64 => true,
            Precision::F32 => f64::from(narrow) == value || value.is_nan(),
            Precision::Bf16 => (f64::from(narrow) == value && narrow.to_bits() & 0xffff == 0) || value.is_nan(),
        };
        if !exact {
            return Err(format!("a checkpoint value {value:e} not held in {precision:?}"));
        }
        match precision {
            Precision::F64 => out.extend_from_slice(&value.to_le_bytes()),
            Precision::F32 => out.extend_from_slice(&narrow.to_le_bytes()),
            Precision::Bf16 => out.extend_from_slice(&((narrow.to_bits() >> 16) as u16).to_le_bytes()),
        }
    }
    Ok(())
}

/// A checkpoint on the host ([`Snapshot::take`]), which a writer thread writes ([`Writer`]) while
/// the fit goes on.
struct Snapshot {
    /// The progress, as the payload's header and as the readable copy.
    header: Vec<u8>,
    json: Vec<u8>,
    /// Per trainable operator `μ`, `ln σ` and IVON's state (the gradient's momentum, the curvature
    /// estimate and the gradient's second moment), each in the storage the device holds it in
    /// (`Progress::precision`).
    payload: Vec<u8>,
}

impl Snapshot {
    /// The checkpoint of `progress` and `posterior` as they are now; `progress` records the
    /// payload's precision. One operator's arrays are on the host as float64 at a time.
    fn take(progress: &mut Progress, posterior: &DevicePosterior) -> Result<Self, String> {
        let precision = posterior.storages().map(Precision::of);
        progress.precision = Some(precision);
        let bytes = checkpoint_payload_bytes(&progress.shapes, precision).ok_or("a checkpoint too large to address")?;
        let mut payload = Vec::with_capacity(usize::try_from(bytes).map_err(error)?);
        for i in 0..progress.shapes.len() {
            let (mean, log_sd, [momentum, curvature, power]) = posterior.operator(i)?;
            for (array, precision) in [&mean, &log_sd, &momentum, &curvature, &power].into_iter().zip(precision) {
                write_checkpoint_array(&mut payload, array, precision)?;
            }
        }
        Ok(Self { header: serde_json::to_vec(progress).map_err(error)?, json: serde_json::to_vec_pretty(progress).map_err(error)?, payload })
    }

    /// Write the checkpoint atomically: the progress as JSON after its length, then the payload.
    /// The progress alone also goes to the path with extension `json`, readable while the fit
    /// runs.
    fn write(&self, path: &Path) -> Result<(), String> {
        let partial = path.with_extension("partial");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&partial).map_err(error)?);
        file.write_all(&(self.header.len() as u64).to_le_bytes()).map_err(error)?;
        file.write_all(&self.header).map_err(error)?;
        file.write_all(&self.payload).map_err(error)?;
        file.into_inner().map_err(error)?.sync_all().map_err(error)?;
        std::fs::rename(&partial, path).map_err(error)?;
        let partial = path.with_extension("json.partial");
        std::fs::write(&partial, &self.json).map_err(error)?;
        std::fs::rename(&partial, path.with_extension("json")).map_err(error)
    }
}

/// The thread writing a fit's last save: at most one write is in flight, and a save first waits
/// for the one before it, so the files on disk are always one whole save.
#[derive(Default)]
struct Writer {
    pending: Option<JoinHandle<Result<(), String>>>,
}

impl Writer {
    /// Wait for the write in flight, with its outcome.
    fn wait(&mut self) -> Result<(), String> {
        match self.pending.take() {
            Some(job) => job.join().map_err(|_| "the checkpoint writer panicked".to_string())?,
            None => Ok(()),
        }
    }

    /// Start `job` once the write in flight is done.
    fn start(&mut self, job: impl FnOnce() -> Result<(), String> + Send + 'static) -> Result<(), String> {
        self.wait()?;
        self.pending = Some(std::thread::spawn(job));
        Ok(())
    }
}

impl Drop for Writer {
    /// A fit that stops early still finishes the write it started.
    fn drop(&mut self) {
        if let Err(e) = self.wait() {
            log::error!("library checkpoint: {e}");
        }
    }
}

/// Restore a checkpoint of this fit into `posterior`, with IVON's state per operator, or refuse one
/// of another fit.
fn load_checkpoint(path: &Path, expected: &Progress, posterior: &mut Posterior) -> Result<(Progress, Vec<[Array2<f64>; 3]>), String> {
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
    let precision = Precision::of_payload(progress.precision);
    if checkpoint_payload_bytes(&progress.shapes, precision) != Some(payload_bytes)
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
        read_checkpoint_array(&mut reader, &mut posterior.mean[i], precision[0])?;
        read_checkpoint_array(&mut reader, &mut posterior.log_sd[i], precision[1])?;
        let mut next = std::array::from_fn(|_| Array2::zeros(dim));
        for (array, precision) in next.iter_mut().zip(&precision[2..]) {
            read_checkpoint_array(&mut reader, array, *precision)?;
        }
        moments.push(next);
    }
    if reader.read(&mut [0_u8; 1]).map_err(error)? != 0 {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    posterior.active = progress.active.clone();
    Ok((progress, moments))
}

/// A checkpoint of `explanation` read whole: the posterior (means, log standard deviations and
/// active groups) and the start it holds ([`checkpoint_start`]).
fn read_checkpoint(explanation: &Explanation, path: &Path) -> Result<(Posterior, Start), String> {
    #[derive(Deserialize)]
    struct Header {
        tokens: usize,
        shapes: Vec<(usize, usize)>,
        active: Vec<bool>,
        #[serde(default)]
        precision: Option<[Precision; CHECKPOINT_ARRAYS]>,
        step: i32,
        epoch: usize,
    }
    let (header, mut reader, payload_bytes): (Header, _, _) = checkpoint_header(path)?;
    let precision = Precision::of_payload(header.precision);
    let mut posterior = Posterior::new(explanation, header.tokens)?;
    if header.shapes != posterior.mean.iter().map(Array2::dim).collect::<Vec<_>>() || header.active.len() != posterior.active.len() {
        return Err(format!("{}: a checkpoint of another explanation", path.display()));
    }
    if checkpoint_payload_bytes(&header.shapes, precision) != Some(payload_bytes) {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    let mut state = Vec::with_capacity(header.shapes.len());
    for (i, shape) in header.shapes.iter().enumerate() {
        read_checkpoint_array(&mut reader, &mut posterior.mean[i], precision[0])?;
        read_checkpoint_array(&mut reader, &mut posterior.log_sd[i], precision[1])?;
        let mut moments: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::zeros(*shape));
        for (array, precision) in moments.iter_mut().zip(&precision[2..]) {
            read_checkpoint_array(&mut reader, array, *precision)?;
        }
        state.push(moments);
    }
    posterior.active = header.active;
    let start = Start {
        mean: posterior.mean.clone(),
        log_sd: posterior.log_sd.clone(),
        active: posterior.active.clone(),
        state: Some(state),
        steps: u64::try_from(header.step).map_err(error)?,
        epoch: header.epoch,
    };
    Ok((posterior, start))
}

/// The posterior of a fit checkpoint of `explanation` (`OUT/checkpoint.bin`, [`fit`]): its means,
/// log standard deviations and active groups, for reading a fit that is still running.
pub fn checkpoint_posterior(explanation: &Explanation, path: &Path) -> Result<Posterior, String> {
    read_checkpoint(explanation, path).map(|(posterior, _)| posterior)
}

/// Where [`fit_from`] starts in place of `M`: per trainable operator (in `Explanation::trainable`
/// order) the posterior means and log standard deviations, which prior groups are in the
/// explanation (a removed group's entries are zeroed), and, to continue an optimizer, IVON's state
/// per operator as a checkpoint holds it (the gradient's momentum, the curvature estimate and the
/// gradient's second moment) after `steps` steps. Without a
/// state, IVON starts at the curvature at which its standard deviations are the posterior's
/// (`DevicePosterior::new`). `epoch` is the next epoch, whose batches' weight noise the fit draws.
pub struct Start {
    pub mean: Vec<Array2<f64>>,
    pub log_sd: Vec<Array2<f64>>,
    pub active: Vec<bool>,
    pub state: Option<Vec<[Array2<f64>; 3]>>,
    pub steps: u64,
    pub epoch: usize,
}

/// The start a fit checkpoint of `explanation` holds: its posterior, IVON's state, the steps taken
/// and the next epoch. [`fit_from`] continues from it exactly as resuming the checkpoint would.
pub fn checkpoint_start(explanation: &Explanation, path: &Path) -> Result<Start, String> {
    read_checkpoint(explanation, path).map(|(_, start)| start)
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
    fit_from(device, native, explanation, sequences, held, settings, export, checkpoint, prior, None)
}

/// [`fit`] from `start` in place of `M` when one is given (a checkpoint's [`checkpoint_start`], a
/// posterior whose groups no checkpoint has, or a start set from the curvature). A checkpoint of
/// this fit, when one exists, is resumed and `start` is not used.
pub fn fit_from(
    device: &Device,
    native: &OperatorProgram,
    explanation: &Explanation,
    sequences: &[Vec<u32>],
    held: &[Vec<u32>],
    settings: &Settings,
    export: &str,
    checkpoint: Option<&Path>,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
    start: Option<Start>,
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
    let mut scorer = Scorer::new(device, native, explanation, settings)?;
    // The fixed collection: its scored tokens N and its realized families.
    let (mut tokens, mut families) = (0, BTreeMap::new());
    for draw in &draws {
        let experiments = scorer.experiments(draw, sequences)?;
        tokens += experiments.iter().map(|e| length - e.position).sum::<usize>();
        for (family, count) in interchange::census(&experiments, scorer.experiments.variables()) {
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
        precision: None,
    };
    // The fixed held-out subset: the first batch of held-out bases (at least the two a source
    // needs).
    let subset = &held[..settings.batch_sequences.clamp(2, held.len())];
    let mut resumed = None;
    if let Some(start) = start {
        let shapes: Vec<(usize, usize)> = posterior.mean.iter().map(Array2::dim).collect();
        let fits = |arrays: &[Array2<f64>]| arrays.iter().map(Array2::dim).eq(shapes.iter().copied());
        let state_fits = start.state.as_ref().is_none_or(|state| state.len() == shapes.len() && state.iter().zip(&shapes).all(|(m, d)| m.iter().all(|a| a.dim() == *d)));
        if !fits(&start.mean) || !fits(&start.log_sd) || start.active.len() != posterior.active.len() || !state_fits {
            return Err("a start of another explanation".into());
        }
        posterior.mean = start.mean;
        posterior.log_sd = start.log_sd;
        posterior.remove(&(0..start.active.len()).filter(|g| !start.active[*g]).collect::<Vec<_>>());
        progress.active = posterior.active.clone();
        progress.epoch = start.epoch;
        progress.step = i32::try_from(start.steps).map_err(error)?;
        resumed = start.state;
    }
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
    let fresh = resumed.is_none();
    let mut device_posterior = DevicePosterior::new(device, explanation, &posterior, tokens as f64, resumed.as_deref(), u64::try_from(progress.step).map_err(error)?)?;
    drop(resumed);
    if fresh {
        let timed = Instant::now();
        let moments = laplace_start(&mut scorer, &mut posterior, &device_posterior, &draws, sequences, settings, tokens)?;
        device_posterior = DevicePosterior::new(device, explanation, &posterior, tokens as f64, Some(&moments), 0)?;
        log::info!("library Laplace start: {:.1} s", timed.elapsed().as_secs_f64());
    }
    // The curvature estimate averages over one epoch's batches: each batch weighs about once.
    let ivon = Ivon { rate: settings.rate, beta1: settings.beta1, beta2: 1.0 - 1.0 / draws.len() as f64 };
    // Each group's size, whose `½ ln |G|` an active group's variance costs.
    let sizes: Vec<f64> = explanation.groups.iter().map(|g| g.cells.iter().map(|c| (c.rows.len() * c.cols.len()) as f64).sum()).collect();
    // After every save, the posterior-mean artifact goes next to the checkpoint (extension
    // `artifact.bin`), so the current explanation can be read and scored while the fit runs. The
    // fit's thread takes the checkpoint and the means off the device; the writer thread encodes
    // and writes them while the device trains on.
    let mut writer = Writer::default();
    let save = |progress: &mut Progress, posterior: &Posterior, device_posterior: &DevicePosterior, writer: &mut Writer| -> Result<(), String> {
        progress.active = posterior.active.clone();
        progress.seconds = resumed_seconds + started.elapsed().as_secs_f64();
        let Some(path) = checkpoint else { return Ok(()) };
        let snapshot = Snapshot::take(progress, device_posterior)?;
        let (artifact, trainable) = (explanation.artifact.clone(), explanation.trainable.clone());
        let (means, membership, active) = (posterior.means(), posterior.membership.clone(), posterior.active.clone());
        let (path, literals) = (path.to_path_buf(), Literals::of(device));
        writer.start(move || {
            snapshot.write(&path)?;
            let partial = path.with_extension("artifact.partial");
            std::fs::write(&partial, literals.apply(mean_artifact(artifact, &trainable, means, &membership, &active)?)?.to_bytes()?).map_err(error)?;
            std::fs::rename(&partial, path.with_extension("artifact.bin")).map_err(error)
        })
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
        let start = held_out(&mut scorer, explanation, (&posterior, &device_posterior), subset, settings, tokens, prior.as_deref_mut())?;
        let seconds = timed.elapsed().as_secs_f64();
        progress.evaluation_seconds += seconds;
        progress.full_seconds = seconds * held.len() as f64 / subset.len() as f64;
        log::info!("library start: {start:?}");
        progress.start = Some(start);
        device_posterior.values_into(&mut posterior)?;
        progress.prior = prior.as_deref().map(PriorTerm::save).transpose()?;
        save(&mut progress, &posterior, &device_posterior, &mut writer)?;
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
        // Each step's code length of the groups' posteriors, summed on the device and read once
        // the epoch's steps are done.
        let mut code = device_posterior.code_length(&posterior.active, &sizes, &posterior.initial, draws.len())?;
        let (mut datas, mut priors) = (Vec::with_capacity(draws.len()), Vec::with_capacity(draws.len()));
        let mut estimates = Vec::with_capacity(draws.len());
        let (mut data_sum, mut description_sum) = (0.0, 0.0);
        let (mut clean, mut patched) = (Mean::default(), Mean::default());
        for (b, draw) in draws.iter().enumerate() {
            let step_started = Instant::now();
            let batch = draw.batch(sequences)?;
            let experiments = scorer.experiments(draw, sequences)?;
            let key = noise_seed(settings.seed, epoch + 1, b);
            let (bits, mut gradients, factor) = scorer.score_device(&device_posterior, &batch, &experiments, Some(key), true)?;
            let factor = factor.ok_or("no Gauss–Newton factor")?;
            for (e, bits) in experiments.iter().zip(&bits) {
                if e.patch.is_some() { patched.add(bits) } else { clean.add(bits) }
            }
            let scored = bits.iter().map(Vec::len).sum::<usize>();
            let (scale, weight) = batch_weights(draws.len(), tokens);
            let data = scale * LN_2 * bits.iter().flatten().sum::<f64>();
            // `Σ_G KL_G` and the active groups' variances' scales at the posterior the sample was drawn
            // from.
            device_posterior.code_length_into(&mut code, b)?;
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
            datas.push(data);
            priors.push(prior_nats);
            progress.step += 1;
            device_posterior.step(&gradients, weight * LN_2, (&factor.gradient, weight), &ivon)?;
            log::info!("library step {epoch}.{b}: {:.6} bits per scored token, {:.2} s", bits.iter().flatten().sum::<f64>() / scored as f64, step_started.elapsed().as_secs_f64());
        }
        for ((code_length, data), prior_nats) in device_posterior.code_lengths(&code)?.into_iter().zip(datas).zip(priors) {
            let description = code_length + subset_code + explanation.fixed_nats + prior_nats;
            if !description.is_finite() {
                return Err("a nonfinite posterior divergence".into());
            }
            estimates.push(data + description);
            data_sum += data;
            description_sum += description;
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
                let evaluation = held_out(&mut scorer, explanation, (&posterior, &device_posterior), subset, settings, tokens, prior.as_deref_mut())?;
                progress.evaluation_seconds += timed.elapsed().as_secs_f64();
                evaluation
            },
            held_out_full: if progress.evaluation_seconds + progress.full_seconds <= EVALUATION_SHARE * progress.training_seconds {
                let timed = Instant::now();
                let evaluation = held_out(&mut scorer, explanation, (&posterior, &device_posterior), held, settings, tokens, prior.as_deref_mut())?;
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
            let log = checkpoint.map(|path| path.with_extension("removals.jsonl"));
            let evidence = Evidence { draws: &draws, sequences, settings };
            let removal = remove(&mut scorer, &mut device_posterior, &mut posterior, &evidence, explanation, prior.as_deref_mut(), Search::Ranked, log.as_deref())?;
            log::info!("library removal after epoch {epoch}: {} of {} candidates, {} without effect", removal.removed, removal.candidates, removal.dead);
            // The removed groups' entries are exactly zero with `ln σ = −∞`, which the device step
            // leaves alone; the objective left its last trial on the device.
            device_posterior.set_values(&posterior)?;
            progress.done = removal.removed == 0;
            progress.removals.push(removal);
            // The objective changed discretely: convergence is judged afresh.
            progress.previous = None;
        }
        progress.prior = prior.as_deref().map(PriorTerm::save).transpose()?;
        save(&mut progress, &posterior, &device_posterior, &mut writer)?;
    }
    let objective_bits = progress.removals.last().map_or(f64::NAN, |r| r.after_bits);
    let end = held_out(&mut scorer, explanation, (&posterior, &device_posterior), held, settings, tokens, prior.as_deref_mut())?;
    log::info!("library end: {end:?}");
    writer.wait()?;
    let representative = if end.rounded_bits_per_token <= end.mean_bits_per_token { Representative::Rounded } else { Representative::Mean };
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
            representative,
            literals: Literals::of(device),
            active_groups: posterior.active.iter().filter(|a| **a).count(),
            objective_bits,
            seconds: resumed_seconds + started.elapsed().as_secs_f64(),
            epochs: progress.epochs,
            removals: progress.removals,
        },
        posterior,
    })
}

/// The Laplace start of a fresh fit. One pass over the training collection draws every batch's
/// sampled-label factor `u_b` at a weight sample of the unit-information start, and
/// `h = Σ_b u_b ⊙ u_b / N` (the batches' estimates `B / N · u_b ⊙ u_b`, averaged) estimates the
/// Gauss–Newton diagonal per token. Each entry's deviation becomes `σ² = 1 / (N h + 1 / v_G)`, the
/// minimum in `σ` of the data term's Gauss–Newton model `½ N h σ²` plus `KL(q ‖ p)`, instead of the
/// epochs IVON's curvature average needs to fall from the start's `1 / v_G` to `h`. Returns IVON's
/// state per operator: zero momentum, curvature `h`, zero second moment.
fn laplace_start(
    scorer: &mut Scorer,
    posterior: &mut Posterior,
    device_posterior: &DevicePosterior,
    draws: &[Draw],
    sequences: &[Vec<u32>],
    settings: &Settings,
    tokens: usize,
) -> Result<Vec<[Array2<f64>; 3]>, String> {
    let mut curvature: Vec<Array2<f64>> = posterior.mean.iter().map(|m| Array2::zeros(m.dim())).collect();
    for (b, draw) in draws.iter().enumerate() {
        let batch = draw.batch(sequences)?;
        let experiments = scorer.experiments(draw, sequences)?;
        let (_, _, factor) = scorer.score_device(device_posterior, &batch, &experiments, Some(noise_seed(settings.seed, 0, b)), true)?;
        let device = scorer.experiments.models().1.program.device();
        for (op, u) in &factor.ok_or("no Gauss–Newton factor")?.gradient {
            let u = device.download(u).map_err(error)?;
            curvature[scorer.at(*op)?].zip_mut_with(&u, |h, u| *h += u * u);
        }
    }
    let variance: Vec<f64> = posterior.moments().iter().map(|m| if m.count > 0.0 { m.second / m.count } else { 0.0 }).collect();
    let n = tokens as f64;
    for (i, h) in curvature.iter_mut().enumerate() {
        h.mapv_inplace(|square| square / n);
        ndarray::Zip::from(&mut posterior.log_sd[i]).and(&*h).and(&posterior.membership[i]).for_each(|s, h, group| {
            let v = variance[*group as usize];
            if *s != f64::NEG_INFINITY && v > 0.0 {
                *s = -0.5 * (n * h + 1.0 / v).ln();
            }
        });
    }
    Ok(curvature.into_iter().map(|h| [Array2::zeros(h.dim()), h.clone(), Array2::zeros(h.dim())]).collect())
}

/// The removal estimates' [`Curvature`]: one draw of the sampled-label gradient per training batch
/// of the fixed collection, at the mean of `posterior` (module note), each draw's group sums made
/// on the device (`DevicePosterior::add_curvature`).
fn removal_curvature(scorer: &mut Scorer, posterior: &DevicePosterior, draws: &[Draw], sequences: &[Vec<u32>], settings: &Settings) -> Result<Curvature, String> {
    posterior.mean_into(scorer.experiments.explanation_mut())?;
    let mut rng = StdRng::seed_from_u64(noise_seed(settings.seed, 0, draws.len()));
    let mut curvature = Curvature::new(posterior.group_count());
    for draw in draws {
        let batch = draw.batch(sequences)?;
        let experiments = scorer.experiments(draw, sequences)?;
        let targets = scorer.experiments.targets(&batch, &experiments)?;
        let rows: usize = experiments.iter().map(|e| batch.length() - e.position).sum();
        let uniforms: Vec<f64> = (0..rows).map(|_| rng.random::<f64>()).collect();
        // One forward pass at the mean, reversed twice: the divergence's gradient (in bits) and a
        // draw of the Gauss–Newton factor.
        let evaluation = scorer.experiments.evaluate_labelled(&batch, &experiments, Some(&targets), true, Some(&uniforms))?;
        let factor = evaluation.factor.ok_or("no Gauss–Newton factor")?;
        posterior.add_curvature(&factor.gradient, 1.0, &mut curvature)?;
        posterior.add_slope(&evaluation.gradient, LN_2, &mut curvature)?;
    }
    Ok(curvature)
}

/// `E_q[D]` over every training batch in nats, one weight sample per batch from the removal seeds,
/// with the groups `removed` (and the already removed ones) zeroed, on the fixed collection; with
/// `prior`, plus its value at each batch's sample, averaged, and the parameters it sends.
fn expected_divergence(
    scorer: &mut Scorer,
    device_posterior: &mut DevicePosterior,
    posterior: &Posterior,
    draws: &[Draw],
    sequences: &[Vec<u32>],
    removed: &[usize],
    settings: &Settings,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<f64, String> {
    let mut trial = posterior.clone();
    trial.remove(removed);
    // The trial goes to the device once; each batch's weight sample is drawn there.
    device_posterior.set_values(&trial)?;
    let mut prior = prior;
    let (mut bits, mut prior_nats) = (0.0, 0.0);
    for (b, draw) in draws.iter().enumerate() {
        // Removal zeroes entries, so the remaining entries see the same noise as the full posterior.
        let key = noise_seed(settings.seed, 0, b);
        let experiments = scorer.experiments(draw, sequences)?;
        let scored = scorer.score_device(device_posterior, &draw.batch(sequences)?, &experiments, Some(key), false)?.0;
        bits += scored.iter().flatten().sum::<f64>();
        if let Some(prior) = prior.as_deref_mut() {
            prior_nats += prior.sample(&trial, &host_sample(&trial, &prior.operators(), key), false)?.0;
        }
    }
    let cost = prior.as_deref().map_or(Ok(0.0), |p| p.cost(&trial))?;
    Ok(bits * LN_2 + prior_nats / draws.len() as f64 + cost)
}

/// A removal round's fixed evidence: the training batches of the sequences under the fit's
/// settings.
struct Evidence<'a> {
    draws: &'a [Draw],
    sequences: &'a [Vec<u32>],
    settings: &'a Settings,
}

/// The removal step (`library_removal`) of `search` on `posterior`, scored by `F` on the round's
/// fixed `evidence`, compensated in the MLPs it deletes functions of (`library_compensation`),
/// logged to `log`.
fn remove(
    scorer: &mut Scorer,
    device_posterior: &mut DevicePosterior,
    posterior: &mut Posterior,
    evidence: &Evidence,
    explanation: &Explanation,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
    search: Search,
    log: Option<&Path>,
) -> Result<Removal, String> {
    let mut prior = prior;
    let (fixed, Evidence { draws, sequences, settings }) = (explanation.fixed_nats, evidence);
    let compensation = Compensation::new(&mut scorer.experiments, explanation, posterior, sequences, settings.batch_sequences)?;
    let curvature = removal_curvature(scorer, device_posterior, draws, sequences, settings)?;
    let mut objective = |trial: &Posterior| -> Result<f64, String> {
        Ok(expected_divergence(scorer, device_posterior, trial, draws, sequences, &[], settings, prior.as_deref_mut())? + trial.description() + fixed)
    };
    library_removal::round(search, explanation, posterior, Some(&compensation), &curvature, &mut objective, log)
}

/// A removal step run on a posterior outside a fit ([`removal_step`]): the fit's training and
/// held-out sequences and settings, and the search with its log.
pub struct Step<'a> {
    pub sequences: &'a [Vec<u32>],
    pub held: &'a [Vec<u32>],
    pub settings: &'a Settings,
    pub search: Search,
    pub log: Option<&'a Path>,
}

/// One removal step of [`fit`] on `posterior` outside a fit (a checkpoint's): the same training
/// collection, weight noise, compensation and acceptance, with the held-out evaluation before and
/// after.
pub fn removal_step(device: &Device, native: &OperatorProgram, explanation: &Explanation, posterior: &mut Posterior, step: Step) -> Result<(Removal, HeldOut, HeldOut), String> {
    let Step { sequences, held, settings, search, log } = step;
    settings.validate()?;
    let length = sequences.first().map_or(0, Vec::len);
    let draws = draws(sequences.len(), settings.batch_sequences, settings.seed)?;
    let mut scorer = Scorer::new(device, native, explanation, settings)?;
    let mut tokens = 0;
    for draw in &draws {
        tokens += scorer.experiments(draw, sequences)?.iter().map(|e| length - e.position).sum::<usize>();
    }
    let evaluate = |scorer: &mut Scorer, posterior: &Posterior| -> Result<HeldOut, String> {
        let device_posterior = DevicePosterior::new(device, explanation, posterior, tokens as f64, None, 0)?;
        held_out(scorer, explanation, (posterior, &device_posterior), held, settings, tokens, None)
    };
    let before = evaluate(&mut scorer, posterior)?;
    let evidence = Evidence { draws: &draws, sequences, settings };
    let mut device_posterior = DevicePosterior::new(device, explanation, posterior, tokens as f64, None, 0)?;
    let removal = remove(&mut scorer, &mut device_posterior, posterior, &evidence, explanation, None, search, log)?;
    let after = evaluate(&mut scorer, posterior)?;
    Ok((removal, before, after))
}

// ----------------------------------------------------------------------------- the reported artifact

/// The explanation at the posterior mean: each library operator holds `μ`, and the blocks only
/// removed groups touch are absent (no literals). Partial interface blocks remain
/// present, so their zeroed entries still cost ordinary serialized literals.
pub fn posterior_mean(explanation: &Explanation, posterior: &Posterior) -> Result<Artifact, String> {
    mean_artifact(explanation.artifact.clone(), &explanation.trainable, posterior.means(), &posterior.membership, &posterior.active)
}

/// [`posterior_mean`] of an explanation's `artifact` whose `trainable` operators hold `means`, each
/// entry in the group `membership` names, the groups active where `active` says.
fn mean_artifact(mut artifact: Artifact, trainable: &[usize], means: Vec<Array2<f64>>, membership: &[Array2<u32>], active: &[bool]) -> Result<Artifact, String> {
    for ((op, values), membership) in trainable.iter().zip(means).zip(membership) {
        let source = &artifact.program.operators[*op];
        let (rows, cols) = (source.rows.clone(), source.cols.clone());
        let mut present = Array2::from_elem((rows.group_count(), cols.group_count()), false);
        for r in 0..rows.group_count() {
            for c in 0..cols.group_count() {
                present[[r, c]] = rows.range(r).any(|i| cols.range(c).any(|j| active[membership[[i, j]] as usize]));
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
    use crate::{
        import::import_language_model,
        run_check::{layer_nodes, split_sites},
    };

    /// `KL(M_e ‖ P_e)` per scored token in bits for `experiments` on `batch` at the explanation's
    /// host values `theta`, and with `gradient` the gradient of its sum in every trainable
    /// operator, on the host.
    fn score(scorer: &mut Scorer, batch: &Batch, experiments: &[Experiment], theta: &[Array2<f64>], gradient: bool) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>), String> {
        let targets = scorer.experiments.targets(batch, experiments)?;
        scorer.experiments.load(theta)?;
        let evaluation = scorer.experiments.evaluate_resident(batch, experiments, &targets, gradient)?;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err("nonfinite explanation divergence".into());
        }
        if !gradient {
            return Ok((evaluation.bits, Vec::new()));
        }
        let d = scorer.experiments.models().1.program.device();
        let mut gradients = Vec::with_capacity(theta.len());
        for (op, values) in scorer.position.iter().map(|(op, i)| (*op, &theta[*i])) {
            gradients.push(match evaluation.gradient.get(&op) {
                Some(g) => d.download(g).map_err(error)?,
                None => Array2::zeros(values.dim()),
            });
        }
        Ok((evaluation.bits, gradients))
    }

    /// The weight sample of `key` of every trainable operator of `posterior` (the device's draws,
    /// `DevicePosterior::sample_into`), on the host.
    fn sample(posterior: &Posterior, key: u64) -> Vec<Array2<f64>> {
        host_sample(posterior, &(0..posterior.mean.len()).collect::<Vec<_>>(), key).into_values().collect()
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
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 72.0, None, 0).unwrap();
        // Every cut and a read patch must be drawn: each held-out base draws one clean and one
        // patched experiment, and six bases leave some of the families out, so the six token
        // rows are cycled into 96 bases (a family drawn with chance p per experiment is missed
        // with chance (1 - p)^96).
        let sequences: Vec<Vec<u32>> = sequences.iter().cycle().take(96).cloned().collect();
        let evaluation = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings, 72, None).unwrap();
        // Every family the held-out sample drew (which hybrid sizes it holds depends on its seed).
        assert!(evaluation.read_patch.is_some() && evaluation.clean.iter().chain(&evaluation.patched).filter(|b| b.is_some()).count() >= 2);
        for bits in evaluation.clean.iter().chain(&evaluation.patched).chain([&evaluation.read_patch]).flatten() {
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
        for (native, layers, _, _) in [tiny("library_owners", "gelu_tanh"), tiny_qwen3("library_owners_qwen3")] {
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
        }
    }

    #[test]
    fn the_reported_objective_pays_for_the_explanation_s_choices() {
        let (native, layers, _, sequences) = tiny("library_choice_bits", "gelu_tanh");
        let mut explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 72.0, None, 0).unwrap();
        let without = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings, 72, None).unwrap();
        explanation.fixed_nats = 1000_f64.ln();
        let with = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings, 72, None).unwrap();
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
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let theta = sample(&posterior, noise_seed(settings.seed, 0, 0));
        let (_, gradients) = score(&mut scorer, &batch, &experiments, &theta, true).unwrap();
        let at = |name: &str| explanation.trainable.iter().position(|op| explanation.artifact.program.operators[*op].name == name).unwrap();
        let (gate, scale) = (at("library.l1.mlp.gate"), at("library.l0.mlp.tie1.f3.scale"));
        let mut bits = |theta: &[Array2<f64>]| -> f64 {
            score(&mut scorer, &batch, &experiments, theta, false).unwrap().0.iter().flatten().sum()
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
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let theta = sample(&posterior, noise_seed(settings.seed, 0, 0));
        let (_, gradients) = score(&mut scorer, &batch, &experiments, &theta, true).unwrap();
        let at = |name: &str| explanation.trainable.iter().position(|op| explanation.artifact.program.operators[*op].name == name).unwrap();
        let (gate, out) = (at("library.l0.mlp.gate"), at("library.l0.mlp.out"));
        let (row_scale, column_scale) = (at("library.l1.mlp.f7.gate.from_l0_gate4.scale"), at("library.l1.mlp.f7.out.from_l0_4.scale"));
        let mut bits = |theta: &[Array2<f64>]| -> f64 { score(&mut scorer, &batch, &experiments, theta, false).unwrap().0.iter().flatten().sum() };
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
    fn a_key_value_group_shared_across_layers_is_differentiated_through_both_of_its_uses() {
        use crate::library_sharing::{Member, share_query_key, share_value};
        let (native, layers, _, sequences) = tiny_qwen3("library_group_gradient");
        let start = explanation(&native, &layers).unwrap();
        // Layer 1's query heads read layer 0's query-key maps, swapped, and its value map is layer
        // 0's through the output projections' transport.
        let shared = share_query_key(&start, &[Member { layer: 0, group: 0, queries: vec![0, 1] }, Member { layer: 1, group: 0, queries: vec![1, 0] }]).unwrap();
        let explanation = share_value(&shared, (1, 0), (0, 0), 0.7).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let theta = sample(&posterior, noise_seed(settings.seed, 0, 0));
        let (_, gradients) = score(&mut scorer, &batch, &experiments, &theta, true).unwrap();
        let at = |name: &str| explanation.trainable.iter().position(|op| explanation.artifact.program.operators[*op].name == name).unwrap();
        let entries = [
            (at("library.l0.h0.q"), (1, 3)),
            (at("library.l0.h1.q"), (2, 5)),
            (at("library.l0.kv0.k"), (0, 6)),
            (at("library.l1.h0.q_shared_scale"), (0, 0)),
            (at("library.l0.kv0.v"), (3, 2)),
            (at("library.l1.kv0.v_from_l0_kv0.scale"), (0, 0)),
        ];
        let mut bits = |theta: &[Array2<f64>]| -> f64 { score(&mut scorer, &batch, &experiments, theta, false).unwrap().0.iter().flatten().sum() };
        for (i, entry) in entries {
            let h = 1e-5;
            let (mut up, mut down) = (theta.clone(), theta.clone());
            up[i][entry] += h;
            down[i][entry] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!((gradients[i][entry] - central).abs() <= 1e-5 * (1.0 + central.abs()), "shared group gradient {} against {central}", gradients[i][entry]);
        }
    }

    #[test]
    fn a_shared_key_is_differentiated_through_every_query_head_reading_it() {
        let (native, layers, _, sequences) = tiny_qwen3("library_shared_key_gradient");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let theta = sample(&posterior, noise_seed(settings.seed, 0, 0));
        let (_, gradients) = score(&mut scorer, &batch, &experiments, &theta, true).unwrap();
        let (k, _) = key_value(&explanation.artifact.program, 1, 0);
        let i = explanation.trainable.iter().position(|op| *op == k).unwrap();
        let mut bits = |theta: &[Array2<f64>]| -> f64 {
            score(&mut scorer, &batch, &experiments, theta, false).unwrap().0.iter().flatten().sum()
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
    fn a_scoped_explanation_charges_and_runs_its_blocks_alone() {
        let (native, layers, _, sequences) = tiny("library_scoped", "relu");
        let full = explanation(&native, &layers).unwrap();
        // Layer 1's MLP alone (block 3).
        let explanation = scoped(&full, &[3]).unwrap();
        let functions = &full.layers[1].functions;
        assert_eq!(explanation.groups.len(), functions.iter().map(Vec::len).sum::<usize>());
        assert!(explanation.groups.iter().all(|g| g.name.starts_with("library.l1.mlp.")));
        assert!(explanation.trainable.iter().all(|op| explanation.artifact.program.operators[*op].name.starts_with("library.l1.mlp.")));
        assert_eq!(scope(&explanation), vec![false, false, false, true]);
        assert_eq!(explanation.layers[1].functions.len(), functions.len());
        assert!(explanation.layers[0].functions.is_empty() && explanation.layers.iter().all(|l| l.heads.is_empty()));
        assert!(scoped(&full, &[4]).is_err() && scoped(&full, &[]).is_err());
        let settings = settings();
        let device = Device::host();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        assert!(experiments.iter().all(|e| e.explained == [false, false, false, true]));
        assert!(experiments.iter().any(|e| e.patch.is_some()));
        // At its start the scoped library is M in every experiment; weight noise in its block
        // alone costs data, and only its layer's functions are counted.
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let cycled: Vec<Vec<u32>> = sequences.iter().cycle().take(96).cloned().collect();
        let evaluation = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &cycled, &settings, 72, None).unwrap();
        for bits in evaluation.clean.iter().chain(&evaluation.patched).chain([&evaluation.read_patch]).flatten() {
            assert!(bits.abs() < 1e-10, "the scoped start diverges from the model by {bits} bits per token");
        }
        assert!(evaluation.data_bits_per_token > 0.0, "weight noise in the block costs data");
        assert_eq!(evaluation.layers[0].functions, 0);
        assert_eq!(evaluation.layers[1].functions, functions.len());
        // The scoped fit converges with the block's groups alone.
        let (train, held) = sequences.split_at(4);
        let fit = fit(&device, &native, &explanation, train, held, &settings, "tiny", None, None).unwrap();
        assert_eq!(fit.posterior.active.len(), explanation.groups.len());
        assert!(fit.report.objective_bits.is_finite());
    }

    #[test]
    fn the_starting_library_matches_the_model_in_every_experiment() {
        let (native, layers, _, sequences) = tiny("library_experiments", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        // Every cut and a read patch must be drawn: each held-out base draws one clean and one
        // patched experiment, and six bases leave some of the families out, so the six token
        // rows are cycled into 96 bases (a family drawn with chance p per experiment is missed
        // with chance (1 - p)^96).
        let sequences: Vec<Vec<u32>> = sequences.iter().cycle().take(96).cloned().collect();
        let evaluation = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings, 72, None).unwrap();
        // Every family the held-out sample drew (which hybrid sizes it holds depends on its seed).
        assert!(evaluation.read_patch.is_some() && evaluation.clean.iter().chain(&evaluation.patched).filter(|b| b.is_some()).count() >= 2);
        for bits in evaluation.clean.iter().chain(&evaluation.patched).chain([&evaluation.read_patch]).flatten() {
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
    fn the_sampled_label_curvature_predicts_the_data_term_to_second_order() {
        // At the library's start P is M on every experiment, so the data term and its gradient vanish
        // at the mean and its Hessian there is the Gauss–Newton matrix H. Shrinking one MLP output
        // group's mean by ε raises the data term by ½ ε² μ_Gᵀ H μ_G, which the sampled-label draws
        // of the removal estimates measure.
        let (native, layers, _, sequences) = tiny("library_curvature", "gelu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 1000).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 1000.0, None, 0).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let group = *explanation.layers[0].functions[1].last().unwrap();
        let rounds = 96;
        let mut quadratic = 0.0;
        for k in 0..rounds {
            let labels = Settings { seed: settings.seed + 1 + k, ..settings.clone() };
            quadratic += removal_curvature(&mut scorer, &device_posterior, &draws, &sequences, &labels).unwrap().quadratic[group] / rounds as f64;
        }
        let epsilon = 1e-2;
        let mut shrunk = posterior.means();
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        for cell in &explanation.groups[group].cells {
            for &r in &cell.rows {
                for c in cell.cols.clone() {
                    shrunk[position[&cell.operator]][[r, c]] *= 1.0 - epsilon;
                }
            }
        }
        let mut exact = 0.0;
        for draw in &draws {
            let experiments = scorer.experiments(draw, &sequences).unwrap();
            let (bits, _) = score(&mut scorer, &draw.batch(&sequences).unwrap(), &experiments, &shrunk, false).unwrap();
            exact += bits.iter().flatten().sum::<f64>() * LN_2;
        }
        let predicted = 0.5 * epsilon * epsilon * quadratic;
        assert!(exact > 0.0 && (predicted - exact).abs() <= 0.25 * exact, "predicted {predicted:e} nats, exact {exact:e}");
    }

    #[test]
    fn removal_scores_the_fixed_collection() {
        // Removing every group changes neither the experiments nor M's targets: a trial is scored
        // on the collection drawn over M's reads, against M's targets.
        let (native, layers, _, sequences) = tiny("library_removal_evidence", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 2 * sequences.len() * 12).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let removed: Vec<usize> = (0..posterior.active.len()).collect();
        let mut trial = posterior.clone();
        trial.remove(&removed);
        // The reference, built apart from the fit's scoring.
        let (mut reference_bits, mut read_patches) = (0.0, 0);
        for (b, draw) in draws.iter().enumerate() {
            let batch = draw.batch(&sequences).unwrap();
            let experiments = scorer.experiments(draw, &sequences).unwrap();
            read_patches += experiments.iter().filter(|e| matches!(e.patch, Some(Patch::Read { .. }))).count();
            let theta = sample(&trial, noise_seed(settings.seed, 0, b));
            let targets = scorer.experiments.targets(&batch, &experiments).unwrap();
            scorer.experiments.load(&theta).unwrap();
            let reference = scorer.experiments.evaluate_resident(&batch, &experiments, &targets, false).unwrap();
            reference_bits += reference.bits.iter().flatten().sum::<f64>();
        }
        assert!(read_patches > 0, "the collection must hold read patches of removed functions");
        let mut device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 1.0, None, 0).unwrap();
        let actual = expected_divergence(&mut scorer, &mut device_posterior, &posterior, &draws, &sequences, &removed, &settings, None).unwrap();
        let reference = reference_bits * LN_2;
        assert!((actual - reference).abs() < 1e-10 * reference.abs().max(1.0), "removal must score the fixed collection");
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
        // A data gradient per operator, weighted by `scale`, and a Gauss–Newton factor, its square
        // weighted by `square`.
        let (scale, square) = (3.0, 0.2);
        let gradients: Vec<Array2<f64>> = posterior.mean.iter().map(|m| m.mapv(|_| rng.random::<f64>() - 0.5)).collect();
        let factors: Vec<Array2<f64>> = posterior.mean.iter().map(|m| m.mapv(|_| rng.random::<f64>() - 0.5)).collect();
        let up = |values: &[Array2<f64>]| -> BTreeMap<usize, Tensor> { explanation.trainable.iter().zip(values).map(|(op, g)| (*op, device.upload(g.view()).unwrap())).collect() };
        let ivon = Ivon { rate: settings.rate, beta1: settings.beta1, beta2: 0.75 };
        device_posterior.step(&up(&gradients), scale, (&up(&factors), square), &ivon).unwrap();
        let mut stepped = posterior.clone();
        device_posterior.download(&mut stepped).unwrap();
        // The host reference: IVON's first step from momentum zero and the curvature at which the
        // posterior's standard deviations are IVON's.
        let variance: Vec<f64> = posterior.moments().iter().map(|m| m.second / m.count).collect();
        let mut reference = posterior.clone();
        for i in 0..reference.mean.len() {
            for ((r, c), mu) in reference.mean[i].indexed_iter_mut() {
                let delta = 1.0 / (tokens * variance[posterior.membership[i][[r, c]] as usize]);
                let sd = posterior.log_sd[i][[r, c]].exp();
                let h0 = (1.0 / (tokens * sd * sd) - delta).max(0.0);
                let d = square * factors[i][[r, c]] * factors[i][[r, c]] - h0;
                let h = h0 + (1.0 - ivon.beta2) * d + 0.5 * (1.0 - ivon.beta2).powi(2) * d * d / (h0 + delta);
                // A first step's momentum is one gradient, which gives no spread: the filtered
                // gradient is zero and the mean moves by the prior's pull alone.
                *mu -= (ivon.rate * (delta * *mu) / (h + delta)).clamp(-sd, sd);
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
    fn the_device_rounds_the_posterior_as_the_host_does() {
        let (native, layers, _, _) = tiny("library_rounded", "gelu");
        let explanation = explanation(&native, &layers).unwrap();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        for log_sd in &mut posterior.log_sd {
            log_sd.mapv_inplace(|s| s + 6.0 * (rng.random::<f64>() - 0.5));
        }
        posterior.remove(&[0]);
        let device = Device::host();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings()).unwrap();
        device_posterior.rounded_into(scorer.experiments.explanation_mut()).unwrap();
        let rounded = posterior.rounded();
        for (i, op) in explanation.trainable.iter().enumerate() {
            let held = device.download(scorer.experiments.explanation_mut().dense(*op).unwrap()).unwrap();
            assert_eq!(held, rounded.mean[i], "operator {i}");
        }
    }

    #[test]
    fn the_device_code_length_is_the_host_description() {
        let (native, layers, _, _) = tiny("library_code_length", "gelu");
        let explanation = explanation(&native, &layers).unwrap();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        // Spread deviations, so the variances' scales differ, and a removed group.
        let mut rng = StdRng::seed_from_u64(3);
        for log_sd in &mut posterior.log_sd {
            log_sd.mapv_inplace(|s| s + 4.0 * (rng.random::<f64>() - 0.5));
        }
        posterior.remove(&[1]);
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 72.0, None, 0).unwrap();
        let sizes: Vec<f64> = explanation.groups.iter().map(|g| g.cells.iter().map(|c| (c.rows.len() * c.cols.len()) as f64).sum()).collect();
        let mut code = device_posterior.code_length(&posterior.active, &sizes, &posterior.initial, 2).unwrap();
        device_posterior.code_length_into(&mut code, 1).unwrap();
        let variances = device_posterior.variances().unwrap();
        let expected: f64 = device_posterior
            .divergences()
            .unwrap()
            .iter()
            .zip(&sizes)
            .zip(&posterior.active)
            .zip(variances.iter().zip(&posterior.initial))
            .map(|(((d, n), active), (v, initial))| if *active { d + 0.5 * n.ln() + scale_bits(*v, *initial) * LN_2 } else { 0.0 })
            .sum();
        let lengths = device_posterior.code_lengths(&code).unwrap();
        assert_eq!(lengths[0], 0.0, "an untouched step's row stays zero");
        assert!(expected.is_finite() && (lengths[1] - expected).abs() <= 1e-12 * expected.abs(), "{} against {expected}", lengths[1]);
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
        read_checkpoint_array(&mut reader, &mut output, Precision::F64).unwrap();
        assert_eq!(reader.position(), bytes.len() as u64);
        assert_eq!(output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
        assert!(read_checkpoint_array(&mut &bytes[..bytes.len() - 1], &mut output, Precision::F64).is_err());
    }

    #[test]
    fn checkpoint_arrays_in_device_precision_round_trip_exactly_and_refuse_other_values() {
        // Values the device holds in each storage: f32 values (with -inf and a subnormal), and
        // bfloat16 values (f32s whose low 16 bits are zero).
        let f32s: Vec<f64> = (0..9001).map(|i| f64::from((i as f32 - 4500.0) / 7.0)).chain([f64::NEG_INFINITY, f64::from(f32::MIN_POSITIVE / 8.0)]).collect();
        let bf16s: Vec<f64> = f32s.iter().map(|v| f64::from(f32::from_bits((*v as f32).to_bits() & 0xffff_0000))).collect();
        for (values, precision) in [(&f32s, Precision::F32), (&bf16s, Precision::Bf16), (&f32s, Precision::F64)] {
            let array = Array2::from_shape_vec((values.len(), 1), values.clone()).unwrap();
            let mut bytes = Vec::new();
            write_checkpoint_array(&mut bytes, &array, precision).unwrap();
            assert_eq!(bytes.len(), values.len() * precision.bytes());
            let mut back = Array2::zeros(array.dim());
            read_checkpoint_array(&mut &bytes[..], &mut back, precision).unwrap();
            assert_eq!(back.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), array.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        let wide = Array2::from_elem((1, 1), 0.1);
        assert!(write_checkpoint_array(&mut Vec::new(), &wide, Precision::F32).is_err());
        assert!(write_checkpoint_array(&mut Vec::new(), &Array2::from_elem((1, 1), f64::from(1.0_f32 + f32::EPSILON)), Precision::Bf16).is_err());
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
            precision: None,
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
        let (wide, device) = ([Precision::F64; CHECKPOINT_ARRAYS], [Precision::F32, Precision::F32, Precision::Bf16, Precision::F32, Precision::F32]);
        assert_eq!(checkpoint_payload_bytes(&[(2, 3), (0, 2), (3, 1)], wide), Some(9 * 40));
        assert_eq!(checkpoint_payload_bytes(&[(2, 3), (0, 2), (3, 1)], device), Some(9 * 18));
        assert_eq!(checkpoint_payload_bytes(&[(usize::MAX, usize::MAX)], wide), None);
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
    fn a_fit_from_the_start_a_checkpoint_holds_is_the_fit_resumed_from_it() {
        let (native, layers, _, sequences) = tiny("library_fit_from", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let (train, held) = sequences.split_at(4);
        let finished_path = std::env::temp_dir().join(format!("library_fit_from_{}.bin", std::process::id()));
        let finished = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&finished_path), None).unwrap();
        // The finished fit's checkpoint reopened, so that it trains again from its posterior and
        // IVON's state, with convergence judged afresh.
        let bytes = std::fs::read(&finished_path).unwrap();
        let length = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
        header["done"] = serde_json::Value::Bool(false);
        header["previous"] = serde_json::Value::Null;
        let header = serde_json::to_vec(&header).unwrap();
        let path = std::env::temp_dir().join(format!("library_fit_from_reopened_{}.bin", std::process::id()));
        std::fs::write(&path, [&(header.len() as u64).to_le_bytes()[..], &header, &bytes[8 + length..]].concat()).unwrap();
        let start = checkpoint_start(&explanation, &path).unwrap();
        assert_eq!(start.epoch, finished.report.epochs.len());
        let resumed = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&path), None).unwrap();
        let started = fit_from(&device, &native, &explanation, train, held, &settings, "tiny", None, None, Some(start)).unwrap();
        for file in [&finished_path, &path] {
            for extension in ["bin", "json", "artifact.bin", "removals.jsonl"] {
                let written = file.with_extension(extension);
                if written.exists() {
                    std::fs::remove_file(written).unwrap();
                }
            }
        }
        let trained = &resumed.report.epochs[finished.report.epochs.len()..];
        assert!(!trained.is_empty(), "the reopened checkpoint trains");
        assert_eq!(trained.len(), started.report.epochs.len());
        for (a, b) in trained.iter().zip(&started.report.epochs) {
            assert_eq!(a.epoch, b.epoch);
            assert_eq!(a.objective_bits.to_bits(), b.objective_bits.to_bits(), "epoch {}", a.epoch);
        }
        assert_eq!(started.posterior.mean, resumed.posterior.mean);
        assert_eq!(started.posterior.log_sd, resumed.posterior.log_sd);
        assert_eq!(started.posterior.active, resumed.posterior.active);
        // A start of another explanation is refused.
        let other = Start { mean: Vec::new(), log_sd: Vec::new(), active: vec![true], state: None, steps: 0, epoch: 0 };
        assert!(fit_from(&device, &native, &explanation, train, held, &settings, "tiny", None, None, Some(other)).is_err());
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
    fn the_scale_reference_survives_a_warm_start() {
        // A warm start from values three times M's keeps every group's reference at M's mean
        // square, so each active group's scale sends the exponent round(log2 9) = 3 and the
        // description grows by the difference of their Elias δ lengths; the start at M sends 0.
        let (native, layers, _, _) = tiny("library_scale_reference", "gelu_tanh");
        let start = explanation(&native, &layers).unwrap();
        let mut moved = start.artifact.clone();
        for &op in &start.trainable {
            let old = Arc::clone(&moved.program.operators[op]);
            let values = old.matrix() * 3.0;
            let precision = exact_precision(values.iter().copied()).unwrap();
            moved.program.operators[op] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).unwrap());
        }
        let warm = crate::library_sharing::warm(&start, &moved).unwrap();
        assert_eq!(warm.reference, start.reference);
        let (cold, hot) = (Posterior::new(&start, 1000).unwrap(), Posterior::new(&warm, 1000).unwrap());
        assert_eq!(hot.initial, cold.initial);
        let zero = scale_bits(1.0, 1.0);
        for (g, (c, h)) in cold.moments().iter().zip(hot.moments()).enumerate() {
            assert_eq!(scale_bits(c.second / c.count, cold.initial[g]), zero, "{}", start.groups[g].name);
            assert_eq!(scale_bits(h.second / h.count, hot.initial[g]), scale_bits(9.0, 1.0), "{}", start.groups[g].name);
        }
    }

    #[test]
    fn every_candidate_is_asked_the_native_questions() {
        // A warm start from moved values, and a tied candidate built from it, are asked the native
        // questions: bitwise the same experiments and targets as the start at M.
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
        let scorers: Vec<Scorer> = [&start, &warm, &tied].iter().map(|e| Scorer::new(&device, &native, e, &settings).unwrap()).collect();
        for (b, draw) in draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap().iter().enumerate() {
            let batch = draw.batch(&sequences).unwrap();
            let experiments: Vec<Vec<Experiment>> = scorers.iter().map(|s| s.experiments(draw, &sequences).unwrap()).collect();
            assert!(experiments.iter().all(|e| *e == experiments[0]), "batch {b}: the experiments");
            let made: Vec<Vec<(Array2<f64>, Vec<f64>)>> = scorers.iter().map(|s| s.experiments.targets(&batch, &experiments[0]).unwrap().host(&device).unwrap()).collect();
            assert!(made.iter().all(|m| *m == made[0]), "batch {b}: the targets");
        }
    }

    #[test]
    fn unequal_batches_weigh_every_scored_token_once() {
        // Six sequences in batches of four and two: the batches' data terms, weighed as the fit
        // weighs them, average to the collection's, and their per-token weights to the
        // collection's per token; weighing each batch by its own scored tokens would not.
        let (native, layers, _, sequences) = tiny("library_batch_weights", "gelu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = Settings { batch_sequences: 4, ..settings() };
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let shrunk: Vec<Array2<f64>> = Posterior::new(&explanation, 1000).unwrap().means().into_iter().map(|m| m * 0.9).collect();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batches: Vec<(f64, usize)> = draws
            .iter()
            .map(|draw| {
                let experiments = scorer.experiments(draw, &sequences).unwrap();
                let (bits, _) = score(&mut scorer, &draw.batch(&sequences).unwrap(), &experiments, &shrunk, false).unwrap();
                (bits.iter().flatten().sum::<f64>() * LN_2, bits.iter().map(Vec::len).sum())
            })
            .collect();
        assert_eq!(batches.len(), 2);
        let tokens: usize = batches.iter().map(|b| b.1).sum();
        let collection: f64 = batches.iter().map(|b| b.0).sum();
        let (scale, weight) = batch_weights(draws.len(), tokens);
        let mean = |w: &dyn Fn(usize) -> f64| batches.iter().map(|&(d, n)| w(n) * d).sum::<f64>() / batches.len() as f64;
        assert!(collection > 0.0 && (mean(&|_| scale) - collection).abs() <= 1e-12 * collection);
        assert!((mean(&|_| weight) - collection / tokens as f64).abs() <= 1e-12 * collection / tokens as f64);
        let own = mean(&|n| tokens as f64 / n as f64);
        assert!((own - collection).abs() > 1e-3 * collection, "{own} against {collection}: the batches' shares of the tokens differ");
    }

    #[test]
    fn held_out_interventions_are_drawn_independently_of_the_training_batches() {
        // Training batch b and held-out batch b, one patched experiment per base: their positions
        // agree at the chance rate 1/12 of two independent uniform positions over 12 tokens, and
        // whole interventions (hybrid, patch and position) agree as often as those of training
        // batches b and b + 1, which are independent by construction.
        let (native, layers, _, sequences) = tiny("library_held_out_seed", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let many: Vec<Vec<u32>> = sequences.iter().cycle().take(2000).cloned().collect();
        let patched = |experiments: Vec<Experiment>| -> Vec<Experiment> { experiments.into_iter().filter(|e| e.patch.is_some()).collect() };
        let training: Vec<Vec<Experiment>> = draws(many.len(), settings.batch_sequences, settings.seed).unwrap().iter().map(|d| patched(scorer.experiments(d, &many).unwrap())).collect();
        let held: Vec<Vec<Experiment>> = held_out_experiments(&scorer, &many, &settings).unwrap().into_iter().map(|(_, e)| patched(e)).collect();
        let pairs = |a: &[Vec<Experiment>], b: &[Vec<Experiment>]| -> Vec<(Experiment, Experiment)> {
            a.iter().zip(b).flat_map(|(x, y)| x.iter().cloned().zip(y.iter().cloned())).collect()
        };
        let aligned = pairs(&training, &held);
        let shifted = pairs(&training, &training[1..]);
        let n = aligned.len() as f64;
        let positions = aligned.iter().filter(|(a, b)| a.position == b.position).count() as f64;
        let (mean, sd) = (n / 12.0, (n / 12.0 * 11.0 / 12.0).sqrt());
        assert!((positions - mean).abs() <= 5.0 * sd, "{positions} of {n} positions agree against {mean} ± {sd}");
        let same = |pairs: &[(Experiment, Experiment)]| pairs.iter().filter(|(a, b)| a.explained == b.explained && a.patch == b.patch && a.position == b.position).count() as f64;
        let (a, s) = (same(&aligned), same(&shifted));
        assert!((a - s).abs() <= 5.0 * (a + s + 1.0).sqrt(), "{a} held-out interventions repeat training ones, against {s} between training batches");
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
        let scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let held: Vec<Vec<u32>> = sequences.iter().cycle().take(2000).cloned().collect();
        let sampled = held_out_experiments(&scorer, &held, &settings).unwrap();
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for (draw, experiments) in &sampled {
            assert_eq!(experiments.len(), 2 * draw.bases.len());
            for (family, count) in interchange::census(experiments, scorer.experiments.variables()) {
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
