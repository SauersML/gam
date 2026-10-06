//! Measured interventions on native language models, for an investigator that writes a model's
//! counterfactual operating manual (#2951): what rule the model follows, what information the rule
//! uses, where the native weights implement it, and how changing those weights changes the rule.
//!
//! A [`Session`] holds named native models (an imported checkpoint's program with every attention
//! and MLP output its own node, [`split_sites`]) and answers [`Request`]s with measured runs only:
//! no linearized or frozen attribution enters any answer.
//!
//! * Sites. The residual stream entering layer `l` (`l = L` is the final residual), the residual
//!   after layer `l`'s attention, layer `l`'s attention output and MLP output (both in the
//!   residual's coordinates), head `h`'s attention read `z_h` (the input of its output map), and
//!   layer `l`'s MLP activations (the input of its down map). Residual states and block outputs
//!   are the same variables in every model of one architecture, so a value moves between models.
//! * Patches. A site's value at chosen rows is replaced by the value the same model computes on a
//!   source sequence (or another model on the same or a source sequence), by zero, by a multiple
//!   of itself, by its mean over reference sequences, or shifted by a vector; restricted to chosen
//!   coordinates, or to its component along one direction `d` (`v ← v + ((t − v)·d̂) d̂`).
//! * Native parameter edits. `W(α) = W + Σ_c (α_c − 1) P_c` on the original stored matrices, with
//!   components `P_c`: a whole operator, its rows or columns, a head's output map (its write, so
//!   `α` scales the head's contribution exactly), an MLP neuron's down column (its write), the
//!   difference from another model's operator (`P = W − W_ref`, or chosen singular components of
//!   it), the part of an operator that reads (`W v vᵀ`) or writes (`u uᵀ W`) one direction, or a
//!   registered component (`Request::Register`, `Request::RegisterVpd`): an id naming the
//!   component's uses, each a low-rank part `P = L R` of one stored operator. A component the
//!   program holds at several operators (a VPD subcomponent of a projection the program splits per
//!   head, one function called at several sites) has one use per operator, and its edit applies
//!   `(α − 1) P` at every use together.
//!   `α = 1` is the model itself, exactly.
//! * Crossed interventions. For inputs `x₀, x₁` and interventions `a₀, a₁`,
//!   `Γ = [y(x₁, a₁) − y(x₀, a₁)] − [y(x₁, a₀) − y(x₀, a₀)]`: how much an intervention changes the
//!   effect of an input distinction on the response `y` (a target token's log-probability, or its
//!   difference from another token's). A component that only represents the distinction leaves
//!   `Γ` at zero when edited; a component the response uses to act on it does not.
//! * Weight differences between two models of one architecture: per operator the norms and
//!   singular values of `W_a − W_b`, and its leading singular directions read in token terms
//!   (a direction a map reads from the residual against the token embeddings through the layer's
//!   norm gain, a direction written to the residual through the final norm gain and the
//!   unembedding: the direct path only, labelled as such).
//!
//! Every run is the host's float64 execution of the program (`OperatorProgram::execute_edited`).

use crate::import::{hugging_face_language_model, import_language_model};
use crate::operator_program::{FamilyInputs, Node, OperatorBody, OperatorProgram, SequenceLayout, SlotValues, exact_precision};
use crate::run_check::{LayerNodes, layer_nodes, split_sites};
use crate::tiled_attention;
use gam_math::categorical::{categorical_kl_from_logits, log_softmax};
use ndarray::{Array1, Array2, Axis};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

/// A model site whose value an investigator reads or patches (module note).
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Site {
    /// The residual stream entering `layer` (`layer = L`: the final residual, before the final norm).
    Stream { layer: usize },
    /// The residual stream after `layer`'s attention, before its MLP.
    Middle { layer: usize },
    /// `layer`'s attention output, in the residual's coordinates.
    Attention { layer: usize },
    /// Head `head`'s attention read `z_h`, the input of its output map.
    Head { layer: usize, head: usize },
    /// `layer`'s MLP output, in the residual's coordinates.
    Mlp { layer: usize },
    /// `layer`'s MLP activations, the input of its down map.
    Neurons { layer: usize },
}

/// What a patch writes at its rows.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PatchValue {
    /// The value computed on `tokens` (default: the patched sequence itself) by `model` (default:
    /// the patched model under the same edits), at `positions` (default: the patched positions).
    Source {
        #[serde(default)]
        tokens: Option<Vec<u32>>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        positions: Option<Vec<i64>>,
    },
    Zero,
    Scale { factor: f64 },
    /// The site's mean over every position of `sequences`, run by the patched model under the same edits.
    Mean { sequences: Vec<Vec<u32>> },
    /// The value plus `vector`.
    Add { vector: Vec<f64> },
    /// `vector` itself at every patched row.
    Set { vector: Vec<f64> },
}

/// A patch of one site at chosen rows (module note).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Patch {
    pub site: Site,
    /// The patched sequences of the request (default: every one).
    #[serde(default)]
    pub sequences: Option<Vec<usize>>,
    /// The patched positions (negative counts from the end; default: every position).
    #[serde(default)]
    pub positions: Option<Vec<i64>>,
    #[serde(default)]
    pub coordinates: Option<Vec<usize>>,
    #[serde(default)]
    pub direction: Option<Vec<f64>>,
    pub value: PatchValue,
}

/// A native parameter component `P_c` (module note).
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Component {
    Operator { name: String },
    Rows { name: String, rows: Vec<usize> },
    Columns { name: String, columns: Vec<usize> },
    /// Head `head`'s output map `blocks.{layer}.o{head}`.
    Head { layer: usize, head: usize },
    /// Column `index` of `blocks.{layer}.down_proj`.
    Neuron { layer: usize, index: usize },
    /// `W − W_ref` for `reference`'s operator `name`, or the sum of its singular components `components`.
    Difference {
        name: String,
        reference: String,
        #[serde(default)]
        components: Option<Vec<usize>>,
    },
    /// The part of `name` reading the unit direction `direction` (`side = input`: `W d dᵀ`) or
    /// writing it (`side = output`: `d dᵀ W`).
    Direction { name: String, side: Side, direction: Vec<f64> },
    /// A component registered with the session under `id` (module note): `P` at each of its uses.
    Registered { id: String },
}

/// One use of a registered component: `P = L R` added to the stored operator `operator`, with
/// `left` (`L`, the operator's rows × `r`) and `right` (`R`, `r` × its columns) as lists of rows.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentUse {
    pub operator: String,
    pub left: Vec<Vec<f64>>,
    pub right: Vec<Vec<f64>>,
}

/// A registered use, resolved: the operator's index and the factors.
#[derive(Clone, Debug)]
struct Use {
    operator: usize,
    left: Array2<f64>,
    right: Array2<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Input,
    Output,
}

/// Which sites a localization swaps: each layer's attention and MLP, or (with `heads`) also each
/// head and each head's operators; within `layers` when given.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    #[serde(default)]
    pub heads: bool,
    #[serde(default)]
    pub layers: Option<Vec<usize>>,
}

/// One edit `(α − 1) P_c` of `W(α)`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub component: Component,
    pub alpha: f64,
}

/// A finite intervention: native edits and patches applied together.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intervention {
    #[serde(default)]
    pub edits: Vec<Edit>,
    #[serde(default)]
    pub patches: Vec<Patch>,
}

/// A prompt and candidate continuations (each with its end-of-turn token, when the protocol has
/// one): the organisms' behaviour protocol scores each option by the summed log-probability of its
/// tokens after the prompt.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionItem {
    pub prompt: Vec<u32>,
    pub options: Vec<Vec<u32>>,
}

/// A crossed pair of option items: the response is the log-probability margin of option `choice`
/// over option `versus` (indices shared by both items).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionPair {
    pub x0: OptionItem,
    pub x1: OptionItem,
    pub choice: usize,
    pub versus: usize,
}

/// One crossed pair: inputs `x₀`, `x₁` and the response `y` at a position of each.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub x0: Vec<u32>,
    pub x1: Vec<u32>,
    pub target: u32,
    #[serde(default)]
    pub versus: Option<u32>,
    /// Response positions in `x₀` and `x₁` (default: the last of each).
    #[serde(default)]
    pub positions: Option<(i64, i64)>,
}

fn default_top() -> usize {
    10
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    pub model: String,
    pub sequences: Vec<Vec<u32>>,
    /// Reported positions (negative counts from the end; default: the last).
    #[serde(default)]
    pub positions: Option<Vec<i64>>,
    #[serde(default = "default_top")]
    pub top: usize,
    #[serde(default)]
    pub targets: Vec<u32>,
    #[serde(default)]
    pub intervention: Intervention,
    /// Also run the model clean and report KL(clean ‖ intervened) and the clean targets.
    #[serde(default)]
    pub clean: bool,
    #[serde(default)]
    pub record: Vec<Site>,
    #[serde(default)]
    pub full: bool,
    /// Also report the log-probability of the sequence's own next token at each reported
    /// position (and the clean model's, with `clean`).
    #[serde(default)]
    pub next: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Info,
    /// Several requests, answered in order, each `{"ok": ...}` or `{"error": ...}`.
    Batch { requests: Vec<Request> },
    /// Per item, each option's summed log-probability under an intervention.
    Options {
        model: String,
        items: Vec<OptionItem>,
        #[serde(default)]
        intervention: Intervention,
    },
    /// Crossed interventions on option margins.
    CrossedOptions {
        model: String,
        pairs: Vec<OptionPair>,
        #[serde(default)]
        a0: Intervention,
        a1: Intervention,
    },
    /// Which sites carry `model`'s different choices from `reference` on option items: the margin
    /// of `model`'s choice over `reference`'s under single-site activation and weight swaps.
    LocalizeOptions {
        model: String,
        reference: String,
        items: Vec<OptionItem>,
        #[serde(default)]
        weights: bool,
        #[serde(default)]
        scope: Scope,
    },
    /// Single-model localization on option items: per site (scope), the margin of each item's
    /// choice over its runner-up when the site is set to zero or to its mean over reference
    /// sequences at every position, and how many items change their choice.
    SweepOptions {
        model: String,
        items: Vec<OptionItem>,
        #[serde(default)]
        mean_over: Option<Vec<Vec<u32>>>,
        #[serde(default)]
        scope: Scope,
    },
    /// A site's mean value at the last token of prompts `a` and of prompts `b`, the unit
    /// direction of their difference, and each prompt's projection on it (an activation
    /// statistic, to be tested by patches along the direction).
    Directions {
        model: String,
        sites: Vec<Site>,
        a: Vec<Vec<u32>>,
        b: Vec<Vec<u32>>,
    },
    /// Writes `W(α) − W` of every edited operator to `directory/<operator>.f64` (float64,
    /// little-endian, row-major) and lists them with their shapes.
    Delta { model: String, edits: Vec<Edit>, directory: String },
    /// Next-token distributions at chosen positions under an intervention.
    Run(RunRequest),
    Crossed {
        model: String,
        pairs: Vec<Pair>,
        #[serde(default)]
        a0: Intervention,
        a1: Intervention,
    },
    Generate {
        model: String,
        tokens: Vec<u32>,
        steps: usize,
        #[serde(default)]
        edits: Vec<Edit>,
        /// Stop after this token (an end of turn).
        #[serde(default)]
        stop: Option<u32>,
    },
    Attention {
        model: String,
        tokens: Vec<u32>,
        layer: usize,
        head: usize,
        #[serde(default)]
        edits: Vec<Edit>,
        #[serde(default = "default_top")]
        top: usize,
    },
    /// The direct-path token reading of a vector written to the residual, or of a site's write at
    /// one position.
    Unembed {
        model: String,
        #[serde(default)]
        vector: Option<Vec<f64>>,
        #[serde(default)]
        site: Option<Site>,
        #[serde(default)]
        tokens: Option<Vec<u32>>,
        #[serde(default)]
        position: Option<i64>,
        #[serde(default = "default_top")]
        top: usize,
    },
    Difference {
        model: String,
        reference: String,
        #[serde(default)]
        spectrum: bool,
        #[serde(default = "default_top")]
        top: usize,
    },
    Components {
        model: String,
        reference: String,
        name: String,
        #[serde(default = "default_components")]
        count: usize,
        #[serde(default = "default_top")]
        top: usize,
        #[serde(default)]
        vectors: bool,
    },
    /// Which sites carry `model`'s difference from `reference` on these sequences, by measured
    /// swaps of activations and of weights in both directions.
    Localize {
        model: String,
        reference: String,
        sequences: Vec<Vec<u32>>,
        #[serde(default)]
        targets: Option<Vec<u32>>,
        #[serde(default)]
        weights: bool,
        #[serde(default)]
        scope: Scope,
    },
    /// KL(model ‖ reference) per position over sequences, the largest positions listed.
    Scan {
        model: String,
        reference: String,
        sequences: Vec<Vec<u32>>,
        #[serde(default = "default_top")]
        top: usize,
    },
    /// KL(full ‖ truncated) per position: one model's next-token distribution with its whole
    /// context against the same model reading only the last `keep` tokens, the largest listed:
    /// where distant context changes the prediction.
    ContextScan {
        model: String,
        sequences: Vec<Vec<u32>>,
        keep: usize,
        #[serde(default = "default_top")]
        top: usize,
    },
    /// The positions of `sequences` where a site's value is largest along a coordinate or a
    /// direction (its largest-activating examples): an activation readout, not an intervention.
    Activations {
        model: String,
        site: Site,
        sequences: Vec<Vec<u32>>,
        #[serde(default)]
        coordinate: Option<usize>,
        #[serde(default)]
        direction: Option<Vec<f64>>,
        #[serde(default = "default_top")]
        top: usize,
    },
    /// Registers components of `model` (module note), each id with its uses; an id registered
    /// again is replaced.
    Register { model: String, components: BTreeMap<String, Vec<ComponentUse>> },
    /// Registers every subcomponent of VPD's exported decomposition at `decomposition`
    /// (`explanation_battery::load_factors`) as id `{site}:{c}` (`h.0.attn.q_proj:17`), `P_c` the
    /// outer product of row `c` of the site's `U` (its output) and column `c` of its `V` (its
    /// input), with one use per operator of `model` that holds part of the site's projection.
    RegisterVpd { model: String, decomposition: String },
}

fn default_components() -> usize {
    4
}

/// An imported native model with its sites named (module note).
pub struct Native {
    pub program: OperatorProgram,
    pub layers: Vec<LayerNodes>,
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_width: usize,
    pub mlp_width: usize,
    pub vocab: usize,
    /// The bytes one input row's node values take in a run (every node's width, float64).
    pub row_bytes: usize,
    operators: BTreeMap<String, usize>,
    /// The final normed hidden state the logits read; a run stops there and forms the logits only
    /// at the rows a request reads.
    hidden: usize,
    /// The final norm's gain and the unembedding operator (`true`: read transposed, `h A`).
    final_gain: Array1<f64>,
    unembedding: (usize, bool),
    embedding: Option<usize>,
}

impl Native {
    /// A Hugging Face checkpoint directory (`config.json`) or a language-model export (`export.json`).
    pub fn load(path: &Path) -> Result<Self, String> {
        let (program, count) = if path.join("config.json").exists() {
            let text = std::fs::read_to_string(path.join("config.json")).map_err(|e| format!("{}: {e}", path.display()))?;
            let config: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            let count = config["num_hidden_layers"].as_u64().ok_or("config.json: num_hidden_layers")? as usize;
            (hugging_face_language_model(path, 0..count)?.0, count)
        } else {
            let imported = import_language_model(path, 0, 0)?;
            let count = imported.record["config"]["n_layers"].as_u64().ok_or("export.json: config.n_layers")? as usize;
            (imported.program, count)
        };
        Self::new(split_sites(&program)?, count)
    }

    pub fn new(program: OperatorProgram, count: usize) -> Result<Self, String> {
        let layers = layer_nodes(&program, count)?;
        let width_of = |node: usize| program.node_interface(node).map(|i| i.width()).map_err(|e| e.to_string());
        let first = &layers[0];
        let operators = program.operators.iter().enumerate().map(|(i, op)| (op.name.clone(), i)).collect::<BTreeMap<_, _>>();
        let Node::Readout { input: logits, .. } = program.nodes[program.output] else {
            return Err("the program's output is not a readout".into());
        };
        let (normed, unembedding) = match &program.nodes[logits] {
            Node::Transposed { input, operator } => (*input, (*operator, true)),
            Node::Affine { terms, bias: None } if terms.len() == 1 => (terms[0].0, (terms[0].1, false)),
            other => return Err(format!("the logits are not a linear head: {other:?}")),
        };
        // Past the hidden state the program holds only the logits and their readout.
        if program.nodes.len() != program.output + 1 || (normed + 1..program.output).any(|n| n != logits) {
            return Err("nodes other than the logits follow the final hidden state".into());
        }
        let final_gain = match &program.nodes[normed] {
            Node::Affine { terms, .. } if terms.len() == 1 => program.operators[terms[0].1].diagonal().ok_or("the final norm has no diagonal gain")?,
            other => return Err(format!("the head does not read a gained norm: {other:?}")),
        };
        let embedding = operators.get("wte").copied();
        let row_bytes = program.interfaces().map_err(|e| e.to_string())?[..=normed].iter().map(|i| i.width()).sum::<usize>() * std::mem::size_of::<f64>();
        Ok(Self {
            hidden: normed,
            row_bytes,
            width: width_of(first.stream)?,
            heads: first.reads.len(),
            kv_heads: first.keys.len(),
            head_width: width_of(first.reads[0])?,
            mlp_width: width_of(first.active)?,
            vocab: program.declarations.domains.first().ok_or("no token domain")?.size,
            operators,
            final_gain,
            unembedding,
            embedding,
            layers,
            program,
        })
    }

    /// `program` (this model's, or an edit of it) up to the final hidden state.
    pub fn body(&self, program: &OperatorProgram) -> OperatorProgram {
        let mut body = program.clone();
        body.nodes.truncate(self.hidden + 1);
        body.output = self.hidden;
        body
    }

    /// The logits of hidden-state rows through `program`'s unembedding.
    pub fn logits(&self, program: &OperatorProgram, hidden: &Array2<f64>) -> Array2<f64> {
        let a = program.operators[self.unembedding.0].matrix_cow();
        if self.unembedding.1 { hidden.dot(a.as_ref()) } else { hidden.dot(&a.t()) }
    }

    pub fn operator(&self, name: &str) -> Result<usize, String> {
        self.operators.get(name).copied().ok_or_else(|| format!("no operator {name}"))
    }

    pub fn node(&self, site: &Site) -> Result<usize, String> {
        let count = self.layers.len();
        let layer = |l: usize| self.layers.get(l).ok_or_else(|| format!("layer {l} of {count}"));
        Ok(match *site {
            Site::Stream { layer: l } if l == count => self.layers[count - 1].residual,
            Site::Stream { layer: l } => layer(l)?.stream,
            Site::Middle { layer: l } => layer(l)?.attended,
            Site::Attention { layer: l } => layer(l)?.attention,
            Site::Head { layer: l, head } => *layer(l)?.reads.get(head).ok_or_else(|| format!("head {head} of {}", self.heads))?,
            Site::Mlp { layer: l } => layer(l)?.mlp,
            Site::Neurons { layer: l } => layer(l)?.active,
        })
    }

    /// A site's value at one row as a write to the residual: a head's through its output map, the
    /// activations through the down map, every other site as it is.
    fn write_of(&self, site: &Site, value: &[f64]) -> Result<Array1<f64>, String> {
        let v = Array1::from(value.to_vec());
        let through = |name: String| -> Result<Array1<f64>, String> { Ok(self.program.operators[self.operator(&name)?].matrix_cow().dot(&v)) };
        match *site {
            Site::Head { layer, head } => through(format!("blocks.{layer}.o{head}")),
            Site::Neurons { layer } => through(format!("blocks.{layer}.down_proj")),
            Site::Stream { .. } | Site::Middle { .. } | Site::Attention { .. } | Site::Mlp { .. } => Ok(v),
        }
    }

    /// The direct-path logits of a residual write `u`: `W_U (γ_f ⊙ u)`, centred over the vocabulary.
    pub fn unembed(&self, u: &Array1<f64>) -> Array1<f64> {
        let g = &self.final_gain * u;
        let a = self.program.operators[self.unembedding.0].matrix_cow();
        let mut out = if self.unembedding.1 { g.dot(a.as_ref()) } else { a.dot(&g) };
        let mean = out.mean().unwrap_or(0.0);
        out -= mean;
        out
    }

    /// Each token's reading of a direction `r` read from the residual entering a layer through its
    /// norm gain `γ`: `(γ ⊙ r)·e_t / √(mean(e_t²))`.
    fn embedding_reading(&self, gain: &Array1<f64>, r: &Array1<f64>) -> Result<Array1<f64>, String> {
        let wte = self.embedding.ok_or("no token embedding")?;
        let a = self.program.operators[wte].matrix_cow();
        let g = gain * r;
        let scores = g.dot(a.as_ref());
        let norms = a.map_axis(Axis(0), |c| (c.iter().map(|x| x * x).sum::<f64>() / c.len() as f64).sqrt().max(f64::MIN_POSITIVE));
        Ok(scores / norms)
    }

    /// The gain of the norm feeding operator `name` (a q, k, v, c_fc or gate_proj map).
    fn read_gain(&self, name: &str) -> Result<Array1<f64>, String> {
        let layer = name.strip_prefix("blocks.").and_then(|r| r.split_once('.')).ok_or_else(|| format!("{name}: not a block operator"))?.0;
        let attention = ["q", "k", "v"].iter().any(|p| name.rsplit('.').next().is_some_and(|last| last.starts_with(p) && last[1..].parse::<usize>().is_ok()));
        let gain = format!("blocks.{layer}.{}.gain", if attention { "rms1" } else { "rms2" });
        self.program.operators[self.operator(&gain)?].diagonal().ok_or_else(|| format!("{gain}: not diagonal"))
    }
}

fn family(sequences: &[Vec<u32>], vocab: usize) -> Result<(FamilyInputs, Vec<usize>), String> {
    let (mut ids, mut sequence, mut position, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (s, tokens) in sequences.iter().enumerate() {
        if tokens.is_empty() {
            return Err(format!("sequence {s} is empty"));
        }
        if let Some(t) = tokens.iter().find(|t| **t as usize >= vocab) {
            return Err(format!("token {t} beyond the vocabulary of {vocab}"));
        }
        offsets.push(ids.len());
        for (p, t) in tokens.iter().enumerate() {
            ids.push(*t);
            sequence.push(s as u32);
            position.push(p as u32);
        }
    }
    let rows = ids.len();
    Ok((FamilyInputs { rows, slots: vec![SlotValues::Tokens(ids)], layout: Some(SequenceLayout { sequence, position }) }, offsets))
}

/// A position of a sequence of `length` tokens (negative counts from the end).
fn resolve(position: i64, length: usize) -> Result<usize, String> {
    let p = if position < 0 { length as i64 + position } else { position };
    if p < 0 || p as usize >= length {
        return Err(format!("position {position} of a sequence of {length}"));
    }
    Ok(p as usize)
}

fn unit(direction: &[f64]) -> Result<Array1<f64>, String> {
    let d = Array1::from(direction.to_vec());
    let norm = d.dot(&d).sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err("a direction must be finite and nonzero".into());
    }
    Ok(d / norm)
}

/// The `top` largest entries of `scores` with their indices, and the `top` smallest.
fn extremes(scores: &Array1<f64>, top: usize) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    let high = order.iter().take(top).map(|i| (*i, scores[*i])).collect();
    let low = order.iter().rev().take(top).map(|i| (*i, scores[*i])).collect();
    (high, low)
}

fn top_log_probabilities(logits: ndarray::ArrayView1<'_, f64>, top: usize) -> Result<(Vec<f64>, Vec<(usize, f64)>), String> {
    let row: Vec<f64> = logits.to_vec();
    let lp = log_softmax(&row).map_err(|e| format!("{e:?}"))?;
    let (high, _) = extremes(&Array1::from(lp.clone()), top);
    Ok((lp, high))
}

/// A patch resolved to rows and per-row target values.
struct Resolved {
    node: usize,
    rows: Vec<usize>,
    /// Per row the value the patch writes there before restriction (`None`: computed from the
    /// row's own value: a scale or a shift).
    targets: Vec<Option<Array1<f64>>>,
    kind: Local,
    coordinates: Option<Vec<usize>>,
    direction: Option<Array1<f64>>,
}

enum Local {
    Zero,
    Fixed,
    Scale(f64),
    Add(Array1<f64>),
}

/// Per sequence, the logits at its reported positions, and per recorded node the values there.
struct Measured {
    logits: Vec<Vec<Array1<f64>>>,
    recorded: Vec<Vec<Vec<Array1<f64>>>>,
}

pub struct Session {
    pub models: BTreeMap<String, Native>,
    /// The memory a run's node values may take; requests run their sequences in groups within it.
    pub work_bytes: usize,
    /// Thin singular value decompositions of operator differences, by (model, reference, operator).
    differences: HashMap<(String, String, String), Arc<gam_linalg::decompose::Svd>>,
    /// Registered components' uses, by (model, id).
    registered: HashMap<(String, String), Vec<Use>>,
}

impl Session {
    pub fn new(models: BTreeMap<String, Native>, work_bytes: usize) -> Self {
        Self { models, work_bytes, differences: HashMap::new(), registered: HashMap::new() }
    }

    /// Registers the component `id` of `model` with its `uses` (module note), each checked against
    /// its operator's shape; returns the number of uses.
    pub fn register(&mut self, model: &str, id: &str, uses: &[ComponentUse]) -> Result<usize, String> {
        let native = self.model(model)?;
        let matrix = |rows: &[Vec<f64>], what: &str| -> Result<Array2<f64>, String> {
            let width = rows.first().map_or(0, Vec::len);
            if rows.is_empty() || width == 0 || rows.iter().any(|r| r.len() != width) || rows.iter().flatten().any(|v| !v.is_finite()) {
                return Err(format!("{id}: {what} must be a nonempty finite matrix given as rows of one length"));
            }
            Array2::from_shape_vec((rows.len(), width), rows.concat()).map_err(|e| e.to_string())
        };
        let mut resolved = Vec::with_capacity(uses.len());
        for u in uses {
            let (left, right) = (matrix(&u.left, "left")?, matrix(&u.right, "right")?);
            resolved.push(Self::use_of(native, id, &u.operator, left, right)?);
        }
        if resolved.is_empty() {
            return Err(format!("{id}: a component needs a use"));
        }
        let count = resolved.len();
        self.registered.insert((model.to_string(), id.to_string()), resolved);
        Ok(count)
    }

    /// A use `L R` of the stored dense operator `name`, its shape checked.
    fn use_of(native: &Native, id: &str, name: &str, left: Array2<f64>, right: Array2<f64>) -> Result<Use, String> {
        let operator = native.operator(name)?;
        let OperatorBody::Dense { values, .. } = &native.program.operators[operator].body else {
            return Err(format!("{id}: {name} is not a stored dense matrix"));
        };
        if left.ncols() != right.nrows() || (left.nrows(), right.ncols()) != values.dim() {
            return Err(format!("{id}: factors {:?} and {:?} on {name} of {:?}", left.dim(), right.dim(), values.dim()));
        }
        Ok(Use { operator, left, right })
    }

    /// Registers every subcomponent of VPD's decomposition at `decomposition` (`Request::RegisterVpd`):
    /// a site's projection is the operator `blocks.{l}.{kind}` when the program holds it whole, or
    /// the program's per-head (per key-value group) operators `blocks.{l}.{q,k,v,o}{h}` in order,
    /// which split a query, key or value map by rows and the output map by columns; each holds the
    /// matching slice of `u_c v_cᵀ`. Returns the number of subcomponents.
    pub fn register_vpd(&mut self, model: &str, decomposition: &Path) -> Result<usize, String> {
        let factors = crate::explanation_battery::load_factors(decomposition)?;
        let native = self.model(model)?;
        let mut registered = Vec::new();
        for f in &factors {
            let kind = f.name.rsplit('.').next().ok_or_else(|| format!("site {}", f.name))?;
            let whole = format!("blocks.{}.{kind}", f.layer);
            // The operators holding the site's projection and the slice each holds (rows of the
            // output for a split query, key or value map, columns of the input for a split output map).
            let parts: Vec<(String, std::ops::Range<usize>, bool)> = if native.operators.contains_key(&whole) {
                vec![(whole, 0..0, false)]
            } else {
                let letter = kind.chars().next().filter(|_| kind.ends_with("_proj") && kind != "down_proj").ok_or_else(|| format!("no operator for site {}", f.name))?;
                let by_rows = letter != 'o';
                let mut parts = Vec::new();
                let mut offset = 0;
                while let Some(&op) = native.operators.get(&format!("blocks.{}.{letter}{}", f.layer, parts.len())) {
                    let (rows, cols) = native.program.operators[op].matrix_cow().dim();
                    let width = if by_rows { rows } else { cols };
                    parts.push((format!("blocks.{}.{letter}{}", f.layer, parts.len()), offset..offset + width, by_rows));
                    offset += width;
                }
                let total = if by_rows { f.u.ncols() } else { f.v.nrows() };
                if parts.is_empty() || offset != total {
                    return Err(format!("site {}: the program's operators hold {offset} of its {total} coordinates", f.name));
                }
                parts
            };
            for c in 0..f.subcomponents() {
                let (u, v) = (f.u.row(c), f.v.column(c));
                let mut uses = Vec::with_capacity(parts.len());
                for (name, range, by_rows) in &parts {
                    let (u, v) = match (range.is_empty(), by_rows) {
                        (true, _) => (u.to_owned(), v.to_owned()),
                        (false, true) => (u.slice(ndarray::s![range.clone()]).to_owned(), v.to_owned()),
                        (false, false) => (u.to_owned(), v.slice(ndarray::s![range.clone()]).to_owned()),
                    };
                    uses.push(Self::use_of(native, &f.name, name, u.insert_axis(Axis(1)), v.insert_axis(Axis(0)))?);
                }
                registered.push((format!("{}:{c}", f.name), uses));
            }
        }
        let count = registered.len();
        for (id, uses) in registered {
            self.registered.insert((model.to_string(), id), uses);
        }
        Ok(count)
    }

    fn model(&self, name: &str) -> Result<&Native, String> {
        self.models.get(name).ok_or_else(|| format!("no model {name} (have {:?})", self.models.keys().collect::<Vec<_>>()))
    }

    fn difference(&mut self, model: &str, reference: &str, name: &str) -> Result<Arc<gam_linalg::decompose::Svd>, String> {
        let key = (model.to_string(), reference.to_string(), name.to_string());
        if let Some(svd) = self.differences.get(&key) {
            return Ok(svd.clone());
        }
        let delta = self.delta(model, reference, name)?;
        let svd = Arc::new(gam_linalg::decompose::svd(delta.view(), false).map_err(|e| format!("{name}: {e:?}"))?);
        self.differences.insert(key, svd.clone());
        Ok(svd)
    }

    fn delta(&self, model: &str, reference: &str, name: &str) -> Result<Array2<f64>, String> {
        let (a, b) = (self.model(model)?, self.model(reference)?);
        let wa = a.program.operators[a.operator(name)?].matrix_cow();
        let wb = b.program.operators[b.operator(name)?].matrix_cow();
        if wa.dim() != wb.dim() {
            return Err(format!("{name}: shapes {:?} and {:?}", wa.dim(), wb.dim()));
        }
        Ok(wa.as_ref() - wb.as_ref())
    }

    /// The component's operators and its matrix `P_c` at each (module note).
    fn component(&mut self, model: &str, component: &Component) -> Result<Vec<(usize, Array2<f64>)>, String> {
        if let Component::Registered { id } = component {
            let uses = self.registered.get(&(model.to_string(), id.clone())).ok_or_else(|| format!("no registered component {id} of {model}"))?;
            return Ok(uses.iter().map(|u| (u.operator, u.left.dot(&u.right))).collect());
        }
        self.single(model, component).map(|p| vec![p])
    }

    /// A component of one operator: its index and `P_c`.
    fn single(&mut self, model: &str, component: &Component) -> Result<(usize, Array2<f64>), String> {
        let native = self.model(model)?;
        let matrix = |name: &str| -> Result<(usize, Array2<f64>), String> {
            let op = native.operator(name)?;
            match &native.program.operators[op].body {
                OperatorBody::Dense { values, .. } => Ok((op, values.matrix().into_owned())),
                OperatorBody::Identity | OperatorBody::LowRank { .. } | OperatorBody::Diagonal { .. } => Err(format!("{name} is not a stored dense matrix")),
            }
        };
        match component {
            Component::Operator { name } => matrix(name),
            Component::Head { layer, head } => matrix(&format!("blocks.{layer}.o{head}")),
            Component::Rows { name, rows } => {
                let (op, w) = matrix(name)?;
                let mut p = Array2::zeros(w.dim());
                for r in rows {
                    let row = w.row(*r);
                    p.row_mut(*r).assign(&row);
                }
                Ok((op, p))
            }
            Component::Columns { name, columns } => {
                let (op, w) = matrix(name)?;
                let mut p = Array2::zeros(w.dim());
                for c in columns {
                    if *c >= w.ncols() {
                        return Err(format!("{name}: column {c} of {}", w.ncols()));
                    }
                    p.column_mut(*c).assign(&w.column(*c));
                }
                Ok((op, p))
            }
            Component::Neuron { layer, index } => {
                let (op, w) = matrix(&format!("blocks.{layer}.down_proj"))?;
                if *index >= w.ncols() {
                    return Err(format!("neuron {index} of {}", w.ncols()));
                }
                let mut p = Array2::zeros(w.dim());
                p.column_mut(*index).assign(&w.column(*index));
                Ok((op, p))
            }
            Component::Direction { name, side, direction } => {
                let (op, w) = matrix(name)?;
                let d = unit(direction)?;
                let column = d.clone().insert_axis(Axis(1));
                let row = d.insert_axis(Axis(0));
                let p = match side {
                    Side::Input if w.ncols() == column.nrows() => w.dot(&column).dot(&row),
                    Side::Output if w.nrows() == column.nrows() => column.dot(&row).dot(&w),
                    Side::Input | Side::Output => return Err(format!("{name} {:?}: a direction of {} on a {:?} matrix", side, column.nrows(), w.dim())),
                };
                Ok((op, p))
            }
            Component::Difference { name, reference, components } => {
                let op = native.operator(name)?;
                let p = match components {
                    None => self.delta(model, reference, name)?,
                    Some(list) => {
                        let svd = self.difference(model, reference, name)?;
                        let mut p = Array2::zeros((svd.u.nrows(), svd.vt.ncols()));
                        for k in list {
                            if *k >= svd.singular_values.len() {
                                return Err(format!("{name}: component {k} of {}", svd.singular_values.len()));
                            }
                            let u = svd.u.column(*k).to_owned().insert_axis(Axis(1));
                            let v = svd.vt.row(*k).to_owned().insert_axis(Axis(0));
                            p = p + u.dot(&v) * svd.singular_values[*k];
                        }
                        p
                    }
                };
                Ok((op, p))
            }
            Component::Registered { id } => Err(format!("{id}: a registered component is resolved by its uses")),
        }
    }

    /// `model`'s program under `W(α) = W + Σ_c (α_c − 1) P_c`.
    pub fn edited(&mut self, model: &str, edits: &[Edit]) -> Result<OperatorProgram, String> {
        let mut deltas: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        for edit in edits {
            if !edit.alpha.is_finite() {
                return Err("a nonfinite amplitude".into());
            }
            for (op, p) in self.component(model, &edit.component)? {
                let entry = deltas.entry(op).or_insert_with(|| Array2::zeros(p.dim()));
                entry.scaled_add(edit.alpha - 1.0, &p);
            }
        }
        let mut program = self.model(model)?.program.clone();
        for (op, delta) in deltas {
            let mut operator = (*program.operators[op]).clone();
            let OperatorBody::Dense { values, present, .. } = &operator.body else {
                return Err(format!("{}: not dense", operator.name));
            };
            let values = &*values.matrix() + &delta;
            let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
            operator.body = OperatorBody::Dense { values: values.into(), present: present.clone(), precision };
            program.operators[op] = Arc::new(operator);
        }
        Ok(program)
    }

    /// Every node's value up to the final hidden state of `program` on `sequences`, with `patches`
    /// applied as each node is computed.
    fn execute(&mut self, model: &str, program: &OperatorProgram, sequences: &[Vec<u32>], patches: &[Patch], edits: &[Edit]) -> Result<(Vec<Array2<f64>>, Vec<usize>), String> {
        let native = self.model(model)?;
        let (inputs, offsets) = family(sequences, native.vocab)?;
        let body = native.body(program);
        let mut resolved: Vec<Resolved> = Vec::new();
        for patch in patches {
            resolved.push(self.resolve_patch(model, program, sequences, &offsets, patch, edits)?);
        }
        let trace = body
            .execute_edited(&inputs, |node, value, _| {
                for r in resolved.iter().filter(|r| r.node == node) {
                    apply(r, value)?;
                }
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        Ok((trace.values, offsets))
    }

    fn resolve_patch(&mut self, model: &str, program: &OperatorProgram, sequences: &[Vec<u32>], offsets: &[usize], patch: &Patch, edits: &[Edit]) -> Result<Resolved, String> {
        let native = self.model(model)?;
        let node = native.node(&patch.site)?;
        let chosen: Vec<usize> = match &patch.sequences {
            Some(list) => list.clone(),
            None => (0..sequences.len()).collect(),
        };
        let mut rows = Vec::new();
        let mut places = Vec::new();
        for s in chosen {
            let tokens = sequences.get(s).ok_or_else(|| format!("patch of sequence {s} of {}", sequences.len()))?;
            let positions: Vec<usize> = match &patch.positions {
                Some(list) => list.iter().map(|p| resolve(*p, tokens.len())).collect::<Result<_, _>>()?,
                None => (0..tokens.len()).collect(),
            };
            for (i, p) in positions.into_iter().enumerate() {
                rows.push(offsets[s] + p);
                places.push((s, i, p));
            }
        }
        let direction = patch.direction.as_deref().map(unit).transpose()?;
        let (kind, targets) = match &patch.value {
            PatchValue::Zero => (Local::Zero, rows.iter().map(|_| None).collect()),
            PatchValue::Scale { factor } => (Local::Scale(*factor), rows.iter().map(|_| None).collect()),
            PatchValue::Add { vector } => (Local::Add(Array1::from(vector.clone())), rows.iter().map(|_| None).collect()),
            PatchValue::Set { vector } => (Local::Fixed, rows.iter().map(|_| Some(Array1::from(vector.clone()))).collect()),
            PatchValue::Mean { sequences: reference } => {
                let every: Vec<Vec<usize>> = reference.iter().map(|t| (0..t.len()).collect()).collect();
                let measured = self.measure(model, reference, &every, &Intervention { edits: edits.to_vec(), patches: vec![] }, &[node])?;
                let all: Vec<&Array1<f64>> = measured.recorded.iter().flat_map(|r| r[0].iter()).collect();
                let first = all.first().ok_or("no reference rows")?;
                let mean = all.iter().fold(Array1::zeros(first.len()), |acc, v| acc + *v) / all.len() as f64;
                (Local::Fixed, rows.iter().map(|_| Some(mean.clone())).collect())
            }
            PatchValue::Source { tokens, model: source_model, positions } => {
                let source_name = source_model.clone().unwrap_or_else(|| model.to_string());
                let source_program = match source_model {
                    Some(other) => self.model(other)?.program.clone(),
                    None => program.clone(),
                };
                let source_native = self.model(&source_name)?;
                let source_node = source_native.node(&patch.site)?;
                // One source run per distinct source sequence.
                let source_sequences: Vec<Vec<u32>> = match tokens {
                    Some(t) => vec![t.clone()],
                    None => sequences.to_vec(),
                };
                let (inputs, source_offsets) = family(&source_sequences, source_native.vocab)?;
                let trace = source_native.body(&source_program).execute(&inputs, false).map_err(|e| e.to_string())?;
                let values = &trace.values[source_node];
                let mut targets = Vec::with_capacity(places.len());
                for (s, i, p) in &places {
                    let (which, length) = match tokens {
                        Some(t) => (0, t.len()),
                        None => (*s, sequences[*s].len()),
                    };
                    let q = match positions {
                        Some(list) => resolve(*list.get(*i).ok_or("fewer source positions than patched positions")?, length)?,
                        None if *p < length => *p,
                        None => return Err(format!("source of {length} tokens has no position {p}")),
                    };
                    targets.push(Some(values.row(source_offsets[which] + q).to_owned()));
                }
                (Local::Fixed, targets)
            }
        };
        Ok(Resolved { node, rows, targets, kind, coordinates: patch.coordinates.clone(), direction })
    }

    pub fn handle(&mut self, request: &Request) -> Result<Value, String> {
        match request {
            Request::Info => Ok(self.info()),
            Request::Batch { requests } => Ok(Value::Array(
                requests
                    .iter()
                    .map(|r| match self.handle(r) {
                        Ok(v) => json!({"ok": v}),
                        Err(e) => json!({"error": e}),
                    })
                    .collect(),
            )),
            Request::Options { model, items, intervention } => {
                let lp = self.option_log_probabilities(model, items, intervention)?;
                Ok(json!({"items": lp.iter().map(|o| json!({"log_probabilities": o, "choice": argmax(o)})).collect::<Vec<_>>()}))
            }
            Request::CrossedOptions { model, pairs, a0, a1 } => self.crossed_options(model, pairs, a0, a1),
            Request::LocalizeOptions { model, reference, items, weights, scope } => self.localize_options(model, reference, items, *weights, scope),
            Request::Delta { model, edits, directory } => self.delta_files(model, edits, directory),
            Request::SweepOptions { model, items, mean_over, scope } => self.sweep_options(model, items, mean_over.as_deref(), scope),
            Request::Directions { model, sites, a, b } => self.directions(model, sites, a, b),
            Request::Run(r) => self.run(r),
            Request::Crossed { model, pairs, a0, a1 } => self.crossed(model, pairs, a0, a1),
            Request::Generate { model, tokens, steps, edits, stop } => self.generate(model, tokens, *steps, edits, *stop),
            Request::Attention { model, tokens, layer, head, edits, top } => self.attention(model, tokens, *layer, *head, edits, *top),
            Request::Unembed { model, vector, site, tokens, position, top } => self.unembed(model, vector.as_deref(), site.as_ref(), tokens.as_deref(), *position, *top),
            Request::Difference { model, reference, spectrum, top } => self.difference_summary(model, reference, *spectrum, *top),
            Request::Components { model, reference, name, count, top, vectors } => self.components(model, reference, name, *count, *top, *vectors),
            Request::Localize { model, reference, sequences, targets, weights, scope } => self.localize(model, reference, sequences, targets.as_deref(), *weights, scope),
            Request::Scan { model, reference, sequences, top } => self.scan(model, reference, sequences, *top),
            Request::ContextScan { model, sequences, keep, top } => self.context_scan(model, sequences, *keep, *top),
            Request::Activations { model, site, sequences, coordinate, direction, top } => self.activations(model, site, sequences, *coordinate, direction.as_deref(), *top),
            Request::Register { model, components } => {
                let mut uses = 0;
                for (id, list) in components {
                    uses += self.register(model, id, list)?;
                }
                Ok(json!({"components": components.len(), "uses": uses}))
            }
            Request::RegisterVpd { model, decomposition } => Ok(json!({"components": self.register_vpd(model, Path::new(decomposition))?})),
        }
    }

    fn info(&self) -> Value {
        let models: BTreeMap<&String, Value> = self
            .models
            .iter()
            .map(|(name, m)| {
                (
                    name,
                    json!({
                        "layers": m.layers.len(), "width": m.width, "heads": m.heads, "kv_heads": m.kv_heads,
                        "head_width": m.head_width, "mlp_width": m.mlp_width, "vocab": m.vocab,
                        "operators": m.operators.keys().filter(|n| n.starts_with("blocks.")).collect::<Vec<_>>(),
                    }),
                )
            })
            .collect();
        json!({ "models": models })
    }

    /// Logits (and the values of `record` nodes) at each sequence's reported positions under
    /// `intervention`, the sequences run in groups whose node values fit the work budget.
    fn measure(&mut self, model: &str, sequences: &[Vec<u32>], positions: &[Vec<usize>], intervention: &Intervention, record: &[usize]) -> Result<Measured, String> {
        let program = self.edited(model, &intervention.edits)?;
        let per_row = self.model(model)?.row_bytes;
        let rows_allowed = (self.work_bytes / per_row).max(1);
        let mut measured = Measured { logits: Vec::with_capacity(sequences.len()), recorded: Vec::with_capacity(sequences.len()) };
        let mut start = 0;
        while start < sequences.len() {
            let mut end = start + 1;
            let mut rows = sequences[start].len();
            while end < sequences.len() && rows + sequences[end].len() <= rows_allowed {
                rows += sequences[end].len();
                end += 1;
            }
            // Patches name sequences of the whole request; a group sees its own members.
            let patches: Vec<Patch> = intervention
                .patches
                .iter()
                .filter_map(|patch| match &patch.sequences {
                    None => Some(patch.clone()),
                    Some(list) => {
                        let local: Vec<usize> = list.iter().filter(|s| (start..end).contains(*s)).map(|s| s - start).collect();
                        (!local.is_empty()).then(|| Patch { sequences: Some(local), ..patch.clone() })
                    }
                })
                .collect();
            let (values, offsets) = self.execute(model, &program, &sequences[start..end], &patches, &intervention.edits)?;
            let native = self.model(model)?;
            let wanted: Vec<usize> = positions[start..end].iter().zip(&offsets).flat_map(|(list, offset)| list.iter().map(move |p| offset + p)).collect();
            let logits = native.logits(&program, &values[native.hidden].select(Axis(0), &wanted));
            let mut next = 0;
            for (local, list) in positions[start..end].iter().enumerate() {
                let offset = offsets[local];
                measured.logits.push(list.iter().map(|_| {
                    next += 1;
                    logits.row(next - 1).to_owned()
                }).collect());
                measured.recorded.push(record.iter().map(|node| list.iter().map(|p| values[*node].row(offset + p).to_owned()).collect()).collect());
            }
            start = end;
        }
        Ok(measured)
    }

    fn logits_at(&mut self, model: &str, sequences: &[Vec<u32>], positions: &[Vec<usize>], intervention: &Intervention) -> Result<Vec<Vec<Array1<f64>>>, String> {
        Ok(self.measure(model, sequences, positions, intervention, &[])?.logits)
    }

    fn run(&mut self, r: &RunRequest) -> Result<Value, String> {
        let RunRequest { model, sequences, positions, top, targets, intervention, clean, record, full, next } = r;
        let (top, clean, full, next) = (*top, *clean, *full, *next);
        let reported: Vec<Vec<usize>> = sequences
            .iter()
            .map(|s| match positions.as_deref() {
                Some(list) => list.iter().map(|p| resolve(*p, s.len())).collect(),
                None => resolve(-1, s.len()).map(|p| vec![p]),
            })
            .collect::<Result<_, _>>()?;
        let nodes = record.iter().map(|site| self.model(model)?.node(site)).collect::<Result<Vec<_>, _>>()?;
        let measured = self.measure(model, sequences, &reported, intervention, &nodes)?;
        let rows = &measured.logits;
        let reference = if clean { Some(self.logits_at(model, sequences, &reported, &Intervention::default())?) } else { None };
        let mut out = Vec::new();
        for (s, list) in reported.iter().enumerate() {
            let mut entries = Vec::new();
            for (i, p) in list.iter().enumerate() {
                let (lp, high) = top_log_probabilities(rows[s][i].view(), top)?;
                let mut entry = json!({
                    "position": p,
                    "top": high.iter().map(|(t, v)| json!([t, v])).collect::<Vec<_>>(),
                    "targets": targets.iter().map(|t| json!([t, lp.get(*t as usize).copied().unwrap_or(f64::NAN)])).collect::<Vec<_>>(),
                });
                if let Some(reference) = &reference {
                    let clean_row = reference[s][i].to_vec();
                    let kl = categorical_kl_from_logits(&clean_row, &rows[s][i].to_vec()).map_err(|e| format!("{e:?}"))?;
                    let clean_lp = log_softmax(&clean_row).map_err(|e| format!("{e:?}"))?;
                    let (_, clean_top) = top_log_probabilities(reference[s][i].view(), top.min(5))?;
                    entry["kl_from_clean_nats"] = json!(kl);
                    entry["clean_targets"] = json!(targets.iter().map(|t| json!([t, clean_lp.get(*t as usize).copied().unwrap_or(f64::NAN)])).collect::<Vec<_>>());
                    entry["clean_top"] = json!(clean_top.iter().map(|(t, v)| json!([t, v])).collect::<Vec<_>>());
                    if let Some(t) = sequences[s].get(p + 1).filter(|_| next) {
                        entry["next"] = json!([t, lp[*t as usize], clean_lp[*t as usize]]);
                    }
                } else if let Some(t) = sequences[s].get(p + 1).filter(|_| next) {
                    entry["next"] = json!([t, lp[*t as usize]]);
                }
                let mut recorded = Vec::new();
                for (k, site) in record.iter().enumerate() {
                    let v = &measured.recorded[s][k][i];
                    let mut r = json!({"site": format!("{site:?}"), "norm": v.dot(v).sqrt()});
                    if matches!(site, Site::Neurons { .. }) {
                        let magnitude = v.mapv(f64::abs);
                        let (high, _) = extremes(&magnitude, top);
                        r["largest"] = json!(high.iter().map(|(i, _)| json!([i, v[*i]])).collect::<Vec<_>>());
                    }
                    if full {
                        r["value"] = json!(v.to_vec());
                    }
                    recorded.push(r);
                }
                if !record.is_empty() {
                    entry["record"] = json!(recorded);
                }
                entries.push(entry);
            }
            out.push(json!({"sequence": s, "positions": entries}));
        }
        Ok(json!({"runs": out}))
    }

    fn crossed(&mut self, model: &str, pairs: &[Pair], a0: &Intervention, a1: &Intervention) -> Result<Value, String> {
        if pairs.is_empty() {
            return Err("no pairs".into());
        }
        let mut sequences = Vec::new();
        let mut positions = Vec::new();
        for pair in pairs {
            let (p0, p1) = pair.positions.unwrap_or((-1, -1));
            positions.push(vec![resolve(p0, pair.x0.len())?]);
            positions.push(vec![resolve(p1, pair.x1.len())?]);
            sequences.push(pair.x0.clone());
            sequences.push(pair.x1.clone());
        }
        let respond = |rows: &[Vec<Array1<f64>>], k: usize, pair: &Pair| -> Result<f64, String> {
            let lp = log_softmax(&rows[k][0].to_vec()).map_err(|e| format!("{e:?}"))?;
            let y = lp[pair.target as usize];
            Ok(match pair.versus {
                Some(v) => y - lp[v as usize],
                None => y,
            })
        };
        let under0 = self.logits_at(model, &sequences, &positions, a0)?;
        let under1 = self.logits_at(model, &sequences, &positions, a1)?;
        let mut out = Vec::new();
        let mut gammas = Vec::new();
        let mut effects = (Vec::new(), Vec::new());
        for (i, pair) in pairs.iter().enumerate() {
            let (y00, y10) = (respond(&under0, 2 * i, pair)?, respond(&under0, 2 * i + 1, pair)?);
            let (y01, y11) = (respond(&under1, 2 * i, pair)?, respond(&under1, 2 * i + 1, pair)?);
            let gamma = (y11 - y01) - (y10 - y00);
            gammas.push(gamma);
            effects.0.push(y10 - y00);
            effects.1.push(y11 - y01);
            out.push(json!({"y_x0_a0": y00, "y_x1_a0": y10, "y_x0_a1": y01, "y_x1_a1": y11, "input_effect_a0": y10 - y00, "input_effect_a1": y11 - y01, "gamma": gamma}));
        }
        Ok(json!({"pairs": out, "gamma": summary(&gammas), "input_effect_a0": summary(&effects.0), "input_effect_a1": summary(&effects.1)}))
    }

    fn generate(&mut self, model: &str, tokens: &[u32], steps: usize, edits: &[Edit], stop: Option<u32>) -> Result<Value, String> {
        let program = self.edited(model, edits)?;
        let vocab = self.model(model)?.vocab;
        let mut sequence = tokens.to_vec();
        let mut chosen = Vec::new();
        for _ in 0..steps {
            let (inputs, _) = family(std::slice::from_ref(&sequence), vocab)?;
            let native = self.model(model)?;
            let trace = native.body(&program).execute(&inputs, false).map_err(|e| e.to_string())?;
            let logits = native.logits(&program, &trace.values[native.hidden].select(Axis(0), &[sequence.len() - 1]));
            let (lp, high) = top_log_probabilities(logits.row(0), 3)?;
            let (next, _) = high[0];
            chosen.push(json!({"token": next, "log_probability": lp[next], "alternatives": high[1..].iter().map(|(t, v)| json!([t, v])).collect::<Vec<_>>()}));
            sequence.push(next as u32);
            if stop == Some(next as u32) {
                break;
            }
        }
        Ok(json!({"generated": chosen}))
    }

    fn attention(&mut self, model: &str, tokens: &[u32], layer: usize, head: usize, edits: &[Edit], top: usize) -> Result<Value, String> {
        let program = self.edited(model, edits)?;
        let native = self.model(model)?;
        let node = native.node(&Site::Head { layer, head })?;
        let Node::Attend { query, key, scale, rotary, causal, .. } = program.nodes[node].clone() else {
            return Err(format!("node {node} is not an attention read"));
        };
        let (inputs, _) = family(&[tokens.to_vec()], native.vocab)?;
        let trace = native.body(&program).execute(&inputs, false).map_err(|e| e.to_string())?;
        let positions: Vec<u32> = (0..tokens.len() as u32).collect();
        let q = tiled_attention::rotate(&trace.values[query], rotary, &positions, false);
        let k = tiled_attention::rotate(&trace.values[key], rotary, &positions, false);
        let weights = tiled_attention::probabilities(q.view(), k.view(), &positions, 0, scale.value(), causal);
        let rows: Vec<Value> = weights
            .outer_iter()
            .enumerate()
            .map(|(p, row)| {
                let (high, _) = extremes(&row.to_owned(), top.min(p + 1));
                json!({"query": p, "sources": high.iter().map(|(s, w)| json!([s, w])).collect::<Vec<_>>()})
            })
            .collect();
        let n = tokens.len();
        let previous = (1..n).map(|p| weights[[p, p - 1]]).sum::<f64>() / (n.max(2) - 1) as f64;
        let first = (1..n).map(|p| weights[[p, 0]]).sum::<f64>() / (n.max(2) - 1) as f64;
        Ok(json!({"rows": rows, "mean_previous_token_weight": previous, "mean_first_token_weight": first}))
    }

    fn unembed(&mut self, model: &str, vector: Option<&[f64]>, site: Option<&Site>, tokens: Option<&[u32]>, position: Option<i64>, top: usize) -> Result<Value, String> {
        let u = match (vector, site, tokens) {
            (Some(v), None, None) => Array1::from(v.to_vec()),
            (None, Some(site), Some(tokens)) => {
                let native = self.model(model)?;
                let (inputs, _) = family(&[tokens.to_vec()], native.vocab)?;
                let trace = native.body(&native.program).execute(&inputs, false).map_err(|e| e.to_string())?;
                let p = resolve(position.unwrap_or(-1), tokens.len())?;
                let value = trace.values[native.node(site)?].row(p).to_vec();
                native.write_of(site, &value)?
            }
            _ => return Err("give either a vector, or a site with tokens (and a position)".into()),
        };
        let native = self.model(model)?;
        if u.len() != native.width {
            return Err(format!("a residual write has {} coordinates, not {}", native.width, u.len()));
        }
        let scores = native.unembed(&u);
        let (high, low) = extremes(&scores, top);
        Ok(json!({"norm": u.dot(&u).sqrt(), "promoted": high, "suppressed": low, "path": "direct (final norm gain and unembedding only)"}))
    }

    fn difference_summary(&mut self, model: &str, reference: &str, spectrum: bool, top: usize) -> Result<Value, String> {
        let names: Vec<String> = self.model(model)?.operators.keys().filter(|n| n.starts_with("blocks.")).cloned().collect();
        let mut rows = Vec::new();
        for name in names {
            if self.model(reference)?.operator(&name).is_err() {
                continue;
            }
            let delta = self.delta(model, reference, &name)?;
            let change = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
            if change == 0.0 {
                continue;
            }
            let b = self.model(reference)?;
            let size = b.program.operators[b.operator(&name)?].matrix_cow().iter().map(|x| x * x).sum::<f64>().sqrt();
            let mut row = json!({"operator": name, "difference_norm": change, "reference_norm": size, "relative": change / size});
            // An MLP map's change concentrated in few neurons shows in its per-neuron norms.
            let neuron_axis = if name.ends_with("down_proj") { Some(Axis(0)) } else if name.ends_with("c_fc") || name.ends_with("gate_proj") { Some(Axis(1)) } else { None };
            if let Some(axis) = neuron_axis {
                let per = delta.map_axis(axis, |v| v.dot(&v).sqrt());
                let (high, _) = extremes(&per, top);
                row["largest_neurons"] = json!(high);
            }
            if spectrum {
                let svd = self.difference(model, reference, &name)?;
                let s = &svd.singular_values;
                row["singular_values"] = json!(s.iter().take(top).collect::<Vec<_>>());
                row["stable_rank"] = json!(s.iter().map(|x| x * x).sum::<f64>() / (s[0] * s[0]));
            }
            rows.push(row);
        }
        rows.sort_by(|x, y| y["relative"].as_f64().unwrap_or(0.0).total_cmp(&x["relative"].as_f64().unwrap_or(0.0)));
        Ok(json!({"changed_operators": rows.len(), "operators": rows}))
    }

    fn components(&mut self, model: &str, reference: &str, name: &str, count: usize, top: usize, vectors: bool) -> Result<Value, String> {
        let svd = self.difference(model, reference, name)?;
        let native = self.model(model)?;
        let writes = name.ends_with("down_proj") || name.rsplit('.').next().is_some_and(|l| l.starts_with('o') && l[1..].parse::<usize>().is_ok());
        let total = svd.singular_values.iter().map(|x| x * x).sum::<f64>();
        let mut out = Vec::new();
        for k in 0..count.min(svd.singular_values.len()) {
            let (u, v) = (svd.u.column(k).to_owned(), svd.vt.row(k).to_owned());
            let mut c = json!({"index": k, "singular_value": svd.singular_values[k], "share_of_squared_norm": svd.singular_values[k].powi(2) / total});
            if writes {
                let scores = native.unembed(&u);
                let (high, low) = extremes(&scores, top);
                c["output_direction_promotes"] = json!(high);
                c["output_direction_suppresses"] = json!(low);
                let magnitude = v.mapv(f64::abs);
                c["input_coordinates"] = json!(extremes(&magnitude, top).0.iter().map(|(i, _)| json!([i, v[*i]])).collect::<Vec<_>>());
            } else {
                let gain = native.read_gain(name)?;
                let scores = native.embedding_reading(&gain, &v)?;
                let (high, low) = extremes(&scores, top);
                c["input_direction_reads_tokens"] = json!(high);
                c["input_direction_reads_negatively"] = json!(low);
                let magnitude = u.mapv(f64::abs);
                c["output_coordinates"] = json!(extremes(&magnitude, top).0.iter().map(|(i, _)| json!([i, u[*i]])).collect::<Vec<_>>());
            }
            if vectors {
                c["left"] = json!(u.to_vec());
                c["right"] = json!(v.to_vec());
            }
            out.push(c);
        }
        Ok(json!({"operator": name, "components": out, "token_readings": "direct path only (embedding through the layer's norm gain, or final norm gain and unembedding)"}))
    }

    /// Per sequence, the response `y` = log-probability of the target (default: `model`'s top
    /// token at the last position) under `reference`, `model`, and each single-site swap.
    fn localize(&mut self, model: &str, reference: &str, sequences: &[Vec<u32>], targets: Option<&[u32]>, weights: bool, scope: &Scope) -> Result<Value, String> {
        let last: Vec<Vec<usize>> = sequences.iter().map(|s| resolve(-1, s.len()).map(|p| vec![p])).collect::<Result<_, _>>()?;
        let none = Intervention::default();
        let upd = self.logits_at(model, sequences, &last, &none)?;
        let base = self.logits_at(reference, sequences, &last, &none)?;
        let targets: Vec<usize> = match targets {
            Some(t) if t.len() == sequences.len() => t.iter().map(|x| *x as usize).collect(),
            Some(t) => return Err(format!("{} targets for {} sequences", t.len(), sequences.len())),
            None => upd.iter().map(|r| extremes(&r[0], 1).0[0].0).collect(),
        };
        let y = |rows: &[Vec<Array1<f64>>]| -> Result<Vec<f64>, String> {
            rows.iter().zip(&targets).map(|(r, t)| log_softmax(&r[0].to_vec()).map(|lp| lp[*t]).map_err(|e| format!("{e:?}"))).collect()
        };
        let (y_model, y_reference) = (y(&upd)?, y(&base)?);
        let sites = self.swap_sites(model, scope)?;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let gap: Vec<f64> = y_model.iter().zip(&y_reference).map(|(a, b)| a - b).collect();
        let mut rows = Vec::new();
        for site in &sites {
            // Into the reference: the site's value from `model` on the same sequence; and the reverse.
            let into = |from: &str| Intervention {
                edits: vec![],
                patches: vec![Patch { site: site.clone(), sequences: None, positions: None, coordinates: None, direction: None, value: PatchValue::Source { tokens: None, model: Some(from.to_string()), positions: None } }],
            };
            let gained = y(&self.logits_at(reference, sequences, &last, &into(model))?)?;
            let kept = y(&self.logits_at(model, sequences, &last, &into(reference))?)?;
            let reproduced: Vec<f64> = gained.iter().zip(&y_reference).map(|(g, b)| g - b).collect();
            let removed: Vec<f64> = y_model.iter().zip(&kept).map(|(m, k)| m - k).collect();
            rows.push(json!({"site": format!("{site:?}"), "kind": "activation",
                "reference_gains_nats": mean(&reproduced), "model_loses_nats": mean(&removed)}));
        }
        if weights {
            for (label, present) in self.swap_groups(model, scope)? {
                let swap = |reference_name: &str| -> Vec<Edit> {
                    present.iter().map(|n| Edit { component: Component::Difference { name: n.clone(), reference: reference_name.to_string(), components: None }, alpha: 0.0 }).collect()
                };
                // The reference with the group's operators set to the model's, and the model with the reference's.
                let gained = y(&self.logits_at(reference, sequences, &last, &Intervention { edits: swap(model), patches: vec![] })?)?;
                let kept = y(&self.logits_at(model, sequences, &last, &Intervention { edits: swap(reference), patches: vec![] })?)?;
                let reproduced: Vec<f64> = gained.iter().zip(&y_reference).map(|(g, b)| g - b).collect();
                let removed: Vec<f64> = y_model.iter().zip(&kept).map(|(m, k)| m - k).collect();
                rows.push(json!({"site": label, "operators": present, "kind": "weights", "reference_gains_nats": mean(&reproduced), "model_loses_nats": mean(&removed)}));
            }
        }
        Ok(json!({
            "targets": targets, "model_log_probability": y_model, "reference_log_probability": y_reference,
            "mean_gap_nats": mean(&gap), "swaps": rows,
        }))
    }

    fn scan(&mut self, model: &str, reference: &str, sequences: &[Vec<u32>], top: usize) -> Result<Value, String> {
        let mut found: Vec<(f64, usize, usize, Vec<(usize, f64)>, Vec<(usize, f64)>)> = Vec::new();
        let (mut total, mut count) = (0.0, 0usize);
        for (s, tokens) in sequences.iter().enumerate() {
            let one = std::slice::from_ref(tokens);
            let positions = vec![(0..tokens.len()).collect::<Vec<_>>()];
            let a = self.logits_at(model, one, &positions, &Intervention::default())?;
            let b = self.logits_at(reference, one, &positions, &Intervention::default())?;
            for (p, (ra, rb)) in a[0].iter().zip(&b[0]).enumerate() {
                let kl = categorical_kl_from_logits(&ra.to_vec(), &rb.to_vec()).map_err(|e| format!("{e:?}"))?;
                total += kl;
                count += 1;
                if found.len() < top || kl > found.last().map_or(f64::NEG_INFINITY, |f| f.0) {
                    let (_, ta) = top_log_probabilities(ra.view(), 3)?;
                    let (_, tb) = top_log_probabilities(rb.view(), 3)?;
                    found.push((kl, s, p, ta, tb));
                    found.sort_by(|x, y| y.0.total_cmp(&x.0));
                    found.truncate(top);
                }
            }
        }
        Ok(json!({
            "mean_kl_nats_per_token": total / count.max(1) as f64, "tokens": count,
            "largest": found.iter().map(|(kl, s, p, ta, tb)| json!({"kl_nats": kl, "sequence": s, "position": p, "model_top": ta, "reference_top": tb})).collect::<Vec<_>>(),
        }))
    }
}

impl Session {
    fn activations(&mut self, model: &str, site: &Site, sequences: &[Vec<u32>], coordinate: Option<usize>, direction: Option<&[f64]>, top: usize) -> Result<Value, String> {
        let direction = direction.map(unit).transpose()?;
        let node = self.model(model)?.node(site)?;
        let mut found: Vec<(f64, usize, usize)> = Vec::new();
        let (mut total, mut squares, mut count) = (0.0, 0.0, 0usize);
        let every: Vec<Vec<usize>> = sequences.iter().map(|t| (0..t.len()).collect()).collect();
        let measured = self.measure(model, sequences, &every, &Intervention::default(), &[node])?;
        for (s, recorded) in measured.recorded.iter().enumerate() {
            for (p, row) in recorded[0].iter().enumerate() {
                let v = match (coordinate, &direction) {
                    (Some(c), None) => *row.get(c).ok_or_else(|| format!("coordinate {c} of {}", row.len()))?,
                    (None, Some(d)) if d.len() == row.len() => row.dot(d),
                    (None, Some(d)) => return Err(format!("a direction of {} at a site of {}", d.len(), row.len())),
                    (None, None) => row.dot(row).sqrt(),
                    (Some(_), Some(_)) => return Err("a coordinate or a direction, not both".into()),
                };
                total += v;
                squares += v * v;
                count += 1;
                if found.len() < top || v > found.last().map_or(f64::NEG_INFINITY, |f| f.0) {
                    found.push((v, s, p));
                    found.sort_by(|x, y| y.0.total_cmp(&x.0));
                    found.truncate(top);
                }
            }
        }
        let mean = total / count.max(1) as f64;
        Ok(json!({
            "mean": mean, "standard_deviation": (squares / count.max(1) as f64 - mean * mean).max(0.0).sqrt(), "tokens": count,
            "largest": found.iter().map(|(v, s, p)| json!({"value": v, "sequence": s, "position": p})).collect::<Vec<_>>(),
        }))
    }

    fn context_scan(&mut self, model: &str, sequences: &[Vec<u32>], keep: usize, top: usize) -> Result<Value, String> {
        if keep == 0 {
            return Err("keep at least one token".into());
        }
        let mut found: Vec<(f64, usize, usize, Vec<(usize, f64)>, Vec<(usize, f64)>)> = Vec::new();
        let (mut total, mut count) = (0.0, 0usize);
        for (s, tokens) in sequences.iter().enumerate() {
            if tokens.len() <= keep {
                continue;
            }
            let all = vec![(0..tokens.len()).collect::<Vec<_>>()];
            let full = self.logits_at(model, std::slice::from_ref(tokens), &all, &Intervention::default())?;
            let windows: Vec<Vec<u32>> = (keep..tokens.len()).map(|p| tokens[p + 1 - keep..=p].to_vec()).collect();
            let last: Vec<Vec<usize>> = windows.iter().map(|_| vec![keep - 1]).collect();
            let truncated = self.logits_at(model, &windows, &last, &Intervention::default())?;
            for (i, p) in (keep..tokens.len()).enumerate() {
                let (rf, rt) = (&full[0][p], &truncated[i][0]);
                let kl = categorical_kl_from_logits(&rf.to_vec(), &rt.to_vec()).map_err(|e| format!("{e:?}"))?;
                total += kl;
                count += 1;
                if found.len() < top || kl > found.last().map_or(f64::NEG_INFINITY, |f| f.0) {
                    let (_, tf) = top_log_probabilities(rf.view(), 3)?;
                    let (_, tt) = top_log_probabilities(rt.view(), 3)?;
                    found.push((kl, s, p, tf, tt));
                    found.sort_by(|x, y| y.0.total_cmp(&x.0));
                    found.truncate(top);
                }
            }
        }
        Ok(json!({
            "mean_kl_nats_per_token": total / count.max(1) as f64, "tokens": count,
            "largest": found.iter().map(|(kl, s, p, tf, tt)| json!({"kl_nats": kl, "sequence": s, "position": p, "full_top": tf, "truncated_top": tt})).collect::<Vec<_>>(),
        }))
    }
}

fn argmax(values: &[f64]) -> Option<usize> {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i)
}

fn summary(v: &[f64]) -> Value {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = if v.len() > 1 { v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) } else { f64::NAN };
    json!({"mean": mean, "standard_error": (var / n).sqrt(), "count": v.len()})
}

impl Session {
    /// Per item, each option's summed log-probability after the prompt (module note).
    /// Per item, each option's summed log-probability after the prompt. A patch's positions
    /// count within the prompt (negative from the prompt's end), so one patch reaches the same
    /// prompt token under every option.
    pub fn option_log_probabilities(&mut self, model: &str, items: &[OptionItem], intervention: &Intervention) -> Result<Vec<Vec<f64>>, String> {
        let mut sequences = Vec::new();
        let mut positions = Vec::new();
        let mut prompts = Vec::new();
        for item in items {
            if item.prompt.is_empty() || item.options.iter().any(Vec::is_empty) {
                return Err("an option item needs a prompt and nonempty options".into());
            }
            for option in &item.options {
                let mut tokens = item.prompt.clone();
                tokens.extend_from_slice(option);
                positions.push((item.prompt.len() - 1..tokens.len() - 1).collect::<Vec<_>>());
                sequences.push(tokens);
                prompts.push(item.prompt.len());
            }
        }
        let mut patches = Vec::new();
        for patch in &intervention.patches {
            match &patch.positions {
                None => patches.push(patch.clone()),
                Some(list) => {
                    let chosen: Vec<usize> = patch.sequences.clone().unwrap_or_else(|| (0..sequences.len()).collect());
                    for s in chosen {
                        let length = *prompts.get(s).ok_or_else(|| format!("patch of sequence {s} of {}", sequences.len()))?;
                        let absolute = list.iter().map(|p| resolve(*p, length).map(|q| q as i64)).collect::<Result<Vec<_>, _>>()?;
                        patches.push(Patch { sequences: Some(vec![s]), positions: Some(absolute), ..patch.clone() });
                    }
                }
            }
        }
        let intervention = Intervention { edits: intervention.edits.clone(), patches };
        let logits = self.logits_at(model, &sequences, &positions, &intervention)?;
        let mut flat = Vec::with_capacity(sequences.len());
        for ((tokens, list), rows) in sequences.iter().zip(&positions).zip(&logits) {
            let mut total = 0.0;
            for (p, row) in list.iter().zip(rows) {
                let lp = log_softmax(&row.to_vec()).map_err(|e| format!("{e:?}"))?;
                total += lp[tokens[p + 1] as usize];
            }
            flat.push(total);
        }
        let mut out = Vec::with_capacity(items.len());
        let mut k = 0;
        for item in items {
            out.push(flat[k..k + item.options.len()].to_vec());
            k += item.options.len();
        }
        Ok(out)
    }

    fn crossed_options(&mut self, model: &str, pairs: &[OptionPair], a0: &Intervention, a1: &Intervention) -> Result<Value, String> {
        let items: Vec<OptionItem> = pairs.iter().flat_map(|p| [p.x0.clone(), p.x1.clone()]).collect();
        let under0 = self.option_log_probabilities(model, &items, a0)?;
        let under1 = self.option_log_probabilities(model, &items, a1)?;
        let margin = |lp: &Vec<f64>, pair: &OptionPair| -> Result<f64, String> {
            match (lp.get(pair.choice), lp.get(pair.versus)) {
                (Some(c), Some(v)) => Ok(c - v),
                _ => Err(format!("options {} and {} of an item with {}", pair.choice, pair.versus, lp.len())),
            }
        };
        let (mut out, mut gammas, mut e0, mut e1) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, pair) in pairs.iter().enumerate() {
            let (y00, y10) = (margin(&under0[2 * i], pair)?, margin(&under0[2 * i + 1], pair)?);
            let (y01, y11) = (margin(&under1[2 * i], pair)?, margin(&under1[2 * i + 1], pair)?);
            let gamma = (y11 - y01) - (y10 - y00);
            gammas.push(gamma);
            e0.push(y10 - y00);
            e1.push(y11 - y01);
            out.push(json!({"y_x0_a0": y00, "y_x1_a0": y10, "y_x0_a1": y01, "y_x1_a1": y11, "gamma": gamma}));
        }
        Ok(json!({"pairs": out, "gamma": summary(&gammas), "input_effect_a0": summary(&e0), "input_effect_a1": summary(&e1)}))
    }

    fn scope_layers(&self, model: &str, scope: &Scope) -> Result<Vec<usize>, String> {
        let count = self.model(model)?.layers.len();
        let layers = scope.layers.clone().unwrap_or_else(|| (0..count).collect());
        if let Some(l) = layers.iter().find(|l| **l >= count) {
            return Err(format!("layer {l} of {count}"));
        }
        Ok(layers)
    }

    /// The sites a single swap reaches: each scoped layer's attention and MLP, and its heads.
    fn swap_sites(&self, model: &str, scope: &Scope) -> Result<Vec<Site>, String> {
        let heads = self.model(model)?.heads;
        let mut sites = Vec::new();
        for layer in self.scope_layers(model, scope)? {
            sites.push(Site::Attention { layer });
            if scope.heads {
                sites.extend((0..heads).map(|head| Site::Head { layer, head }));
            }
            sites.push(Site::Mlp { layer });
        }
        Ok(sites)
    }

    /// Operator groups a weight swap reverts together: per scoped layer its attention maps and its
    /// MLP maps; with heads, per head its query and output maps and per layer its keys and values.
    fn swap_groups(&self, model: &str, scope: &Scope) -> Result<Vec<(String, Vec<String>)>, String> {
        let native = self.model(model)?;
        let mut groups = Vec::new();
        for l in self.scope_layers(model, scope)? {
            let keys_values: Vec<String> = (0..native.kv_heads).flat_map(|g| [format!("blocks.{l}.k{g}"), format!("blocks.{l}.v{g}")]).collect();
            if scope.heads {
                for h in 0..native.heads {
                    groups.push((format!("layer {l} head {h} (q, o)"), vec![format!("blocks.{l}.q{h}"), format!("blocks.{l}.o{h}")]));
                }
                groups.push((format!("layer {l} keys and values"), keys_values));
            } else {
                let mut all: Vec<String> = (0..native.heads).flat_map(|h| [format!("blocks.{l}.q{h}"), format!("blocks.{l}.o{h}")]).collect();
                all.extend(keys_values);
                groups.push((format!("layer {l} attention"), all));
            }
            groups.push((format!("layer {l} MLP"), ["c_fc", "gate_proj", "down_proj"].iter().map(|p| format!("blocks.{l}.{p}")).collect()));
        }
        Ok(groups
            .into_iter()
            .map(|(label, names)| (label, names.into_iter().filter(|n| native.operators.contains_key(n)).collect::<Vec<_>>()))
            .filter(|(_, names)| !names.is_empty())
            .collect())
    }

    fn localize_options(&mut self, model: &str, reference: &str, items: &[OptionItem], weights: bool, scope: &Scope) -> Result<Value, String> {
        let none = Intervention::default();
        let lp_model = self.option_log_probabilities(model, items, &none)?;
        let lp_reference = self.option_log_probabilities(reference, items, &none)?;
        let choices: Vec<(usize, usize)> = lp_model
            .iter()
            .zip(&lp_reference)
            .map(|(m, r)| argmax(m).zip(argmax(r)).ok_or_else(|| "an item without options".to_string()))
            .collect::<Result<_, _>>()?;
        // Items where the two models choose differently carry the difference.
        let differing: Vec<usize> = (0..items.len()).filter(|i| choices[*i].0 != choices[*i].1).collect();
        if differing.is_empty() {
            return Ok(json!({"differing_items": 0, "choices": choices}));
        }
        let subset: Vec<OptionItem> = differing.iter().map(|i| items[*i].clone()).collect();
        let margin = |lp: &[Vec<f64>]| -> Vec<f64> { differing.iter().zip(lp).map(|(i, l)| l[choices[*i].0] - l[choices[*i].1]).collect() };
        let y_model = margin(&differing.iter().map(|i| lp_model[*i].clone()).collect::<Vec<_>>());
        let y_reference = margin(&differing.iter().map(|i| lp_reference[*i].clone()).collect::<Vec<_>>());
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let mut rows = Vec::new();
        for site in self.swap_sites(model, scope)? {
            let into = |from: &str| Intervention {
                edits: vec![],
                patches: vec![Patch { site: site.clone(), sequences: None, positions: None, coordinates: None, direction: None, value: PatchValue::Source { tokens: None, model: Some(from.to_string()), positions: None } }],
            };
            let gained = margin(&self.option_log_probabilities(reference, &subset, &into(model))?);
            let kept = margin(&self.option_log_probabilities(model, &subset, &into(reference))?);
            rows.push(json!({"site": format!("{site:?}"), "kind": "activation",
                "reference_gains_nats": mean(&gained) - mean(&y_reference), "model_loses_nats": mean(&y_model) - mean(&kept),
                "reference_flips": gained.iter().filter(|g| **g > 0.0).count(), "model_flips": kept.iter().filter(|k| **k < 0.0).count()}));
        }
        if weights {
            for (label, names) in self.swap_groups(model, scope)? {
                let swap = |other: &str| -> Vec<Edit> {
                    names.iter().map(|n| Edit { component: Component::Difference { name: n.clone(), reference: other.to_string(), components: None }, alpha: 0.0 }).collect()
                };
                let gained = margin(&self.option_log_probabilities(reference, &subset, &Intervention { edits: swap(model), patches: vec![] })?);
                let kept = margin(&self.option_log_probabilities(model, &subset, &Intervention { edits: swap(reference), patches: vec![] })?);
                rows.push(json!({"site": label, "operators": names, "kind": "weights",
                    "reference_gains_nats": mean(&gained) - mean(&y_reference), "model_loses_nats": mean(&y_model) - mean(&kept),
                    "reference_flips": gained.iter().filter(|g| **g > 0.0).count(), "model_flips": kept.iter().filter(|k| **k < 0.0).count()}));
            }
        }
        Ok(json!({
            "differing_items": differing.len(), "items": items.len(), "choices": choices,
            "mean_margin_model_nats": mean(&y_model), "mean_margin_reference_nats": mean(&y_reference), "swaps": rows,
        }))
    }

    fn sweep_options(&mut self, model: &str, items: &[OptionItem], mean_over: Option<&[Vec<u32>]>, scope: &Scope) -> Result<Value, String> {
        let clean = self.option_log_probabilities(model, items, &Intervention::default())?;
        // Each item's choice and runner-up under the model as it is.
        let ranked: Vec<(usize, usize)> = clean
            .iter()
            .map(|lp| {
                let mut order: Vec<usize> = (0..lp.len()).collect();
                order.sort_by(|a, b| lp[*b].total_cmp(&lp[*a]));
                (order[0], *order.get(1).unwrap_or(&order[0]))
            })
            .collect();
        let margin = |lp: &[Vec<f64>]| -> Vec<f64> { lp.iter().zip(&ranked).map(|(l, (c, v))| l[*c] - l[*v]).collect() };
        let base = margin(&clean);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let sites = self.swap_sites(model, scope)?;
        // Each site's mean over every position of the reference sequences, from one measurement.
        let means: Option<Vec<Array1<f64>>> = match mean_over {
            Some(reference) => {
                let nodes = sites.iter().map(|site| self.model(model)?.node(site)).collect::<Result<Vec<_>, _>>()?;
                let every: Vec<Vec<usize>> = reference.iter().map(|t| (0..t.len()).collect()).collect();
                let measured = self.measure(model, reference, &every, &Intervention::default(), &nodes)?;
                let count = every.iter().map(Vec::len).sum::<usize>().max(1) as f64;
                Some(
                    (0..nodes.len())
                        .map(|k| {
                            let rows: Vec<&Array1<f64>> = measured.recorded.iter().flat_map(|r| r[k].iter()).collect();
                            rows.iter().fold(Array1::zeros(rows.first().map_or(0, |v| v.len())), |acc, v| acc + *v) / count
                        })
                        .collect(),
                )
            }
            None => None,
        };
        let mut rows = Vec::new();
        for (k, site) in sites.into_iter().enumerate() {
            let value = match &means {
                Some(m) => PatchValue::Set { vector: m[k].to_vec() },
                None => PatchValue::Zero,
            };
            let patch = Patch { site: site.clone(), sequences: None, positions: None, coordinates: None, direction: None, value };
            let lp = self.option_log_probabilities(model, items, &Intervention { edits: vec![], patches: vec![patch] })?;
            let after = margin(&lp);
            let flips: Vec<usize> = lp.iter().zip(&ranked).enumerate().filter(|(_, (l, (c, _)))| argmax(l) != Some(*c)).map(|(i, _)| i).collect();
            rows.push(json!({"site": format!("{site:?}"), "margin_change_nats": mean(&after) - mean(&base), "flipped_items": flips,
                "margins": after}));
        }
        Ok(json!({"choices": ranked, "margins": base, "sites": rows}))
    }

    fn directions(&mut self, model: &str, sites: &[Site], a: &[Vec<u32>], b: &[Vec<u32>]) -> Result<Value, String> {
        if a.is_empty() || b.is_empty() {
            return Err("two nonempty groups of prompts".into());
        }
        let nodes = sites.iter().map(|site| self.model(model)?.node(site)).collect::<Result<Vec<_>, _>>()?;
        let group = |session: &mut Session, prompts: &[Vec<u32>]| -> Result<Vec<Vec<Array1<f64>>>, String> {
            let last: Vec<Vec<usize>> = prompts.iter().map(|p| resolve(-1, p.len()).map(|q| vec![q])).collect::<Result<_, _>>()?;
            let measured = session.measure(model, prompts, &last, &Intervention::default(), &nodes)?;
            // Per site, each prompt's value at its last token.
            Ok((0..nodes.len()).map(|k| measured.recorded.iter().map(|r| r[k][0].clone()).collect()).collect())
        };
        let (va, vb) = (group(self, a)?, group(self, b)?);
        let mut out = Vec::new();
        for (k, site) in sites.iter().enumerate() {
            let average = |v: &[Array1<f64>]| v.iter().fold(Array1::<f64>::zeros(v[0].len()), |acc, x| acc + x) / v.len() as f64;
            let difference = average(&va[k]) - average(&vb[k]);
            let norm = difference.dot(&difference).sqrt();
            let direction = if norm > 0.0 { &difference / norm } else { difference.clone() };
            let project = |v: &[Array1<f64>]| v.iter().map(|x| x.dot(&direction)).collect::<Vec<_>>();
            let typical = va[k].iter().chain(&vb[k]).map(|x| x.dot(x).sqrt()).sum::<f64>() / (va[k].len() + vb[k].len()) as f64;
            out.push(json!({"site": format!("{site:?}"), "difference_norm": norm, "typical_norm": typical,
                "projections_a": project(&va[k]), "projections_b": project(&vb[k]), "direction": direction.to_vec()}));
        }
        Ok(json!({"sites": out}))
    }

    fn delta_files(&mut self, model: &str, edits: &[Edit], directory: &str) -> Result<Value, String> {
        let edited = self.edited(model, edits)?;
        let native = self.model(model)?;
        std::fs::create_dir_all(directory).map_err(|e| format!("{directory}: {e}"))?;
        let mut out = Vec::new();
        for (index, (a, b)) in edited.operators.iter().zip(&native.program.operators).enumerate() {
            if Arc::ptr_eq(a, b) {
                continue;
            }
            let delta = a.matrix_cow().as_ref() - b.matrix_cow().as_ref();
            let path = Path::new(directory).join(format!("{}.f64", a.name));
            let bytes: Vec<u8> = delta.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            out.push(json!({"operator": a.name, "index": index, "shape": [delta.nrows(), delta.ncols()], "file": path.display().to_string(),
                "changed_entries": delta.iter().filter(|v| **v != 0.0).count()}));
        }
        Ok(json!({"operators": out}))
    }
}

fn apply(r: &Resolved, value: &mut Array2<f64>) -> Result<(), String> {
    let width = value.ncols();
    for (i, row) in r.rows.iter().enumerate() {
        let current = value.row(*row).to_owned();
        let target = match (&r.kind, &r.targets[i]) {
            (Local::Zero, _) => Array1::zeros(width),
            (Local::Fixed, Some(t)) => t.clone(),
            (Local::Scale(f), _) => &current * *f,
            (Local::Add(a), _) => &current + a,
            (Local::Fixed, None) => return Err("a fixed patch without a value".into()),
        };
        if target.len() != width {
            return Err(format!("a patch value of {} coordinates at a site of {width}", target.len()));
        }
        let mut next = current.clone();
        match (&r.direction, &r.coordinates) {
            (Some(d), None) if d.len() == width => {
                let along = (&target - &current).dot(d);
                next.scaled_add(along, d);
            }
            (Some(d), None) => return Err(format!("a direction of {} coordinates at a site of {width}", d.len())),
            (None, Some(list)) => {
                for c in list {
                    if *c >= width {
                        return Err(format!("coordinate {c} of {width}"));
                    }
                    next[*c] = target[*c];
                }
            }
            (None, None) => next = target,
            (Some(_), Some(_)) => return Err("a patch takes coordinates or a direction, not both".into()),
        }
        value.row_mut(*row).assign(&next);
    }
    Ok(())
}
