//! VPD's slices with intrinsic gates as a library explanation (#2951, the main line on vpd4l):
//! `library_mdl` fits it like any library (snapshot scoring, removal through the whole program,
//! the per-token budget), on the shared verbatim experiments.
//!
//! The start is `vpd_start`'s: per arm, components of VPD's rank-one slices (site, index), each with
//! one gate that reads only its own model's input at its own row, so the explanation is causal and
//! autonomous: no gating network, no all-on pass. A component is on at a row iff its gate's
//! pre-activation is positive, `ln ‖V_bᵀx‖ − ln τ_b` (its own read, on the log scale: a part whose
//! reads are zero is off; shared own gates read the squared norm, below) or `g_bᵀx + c_b − τ_b` (a
//! separate direction), and then all its slices run (`Node::GroupNorm`, `Law::Log`, `Node::Gated`). Its rank, the slices
//! it runs, counts against the per-token budget. The remainder `W − Σ U_i V_iᵀ` is dropped: the
//! explanation does not use it, so it is not charged.
//!
//! A stage gates one way: all its components on their own reads or all on directions (the two
//! build different nodes, a direction gate before the reads it gates, an own gate after them), so
//! a stage mixing the two is refused; different stages of one explanation may differ.
//!
//! A logit read (`Read::Logits`, descent's rot arm's query blocks) gates q components on the spread
//! of their terms in the attention logits over the keys, under each head's pattern with every q
//! and k slice on: each head's rule computes every head's pattern and key moments
//! (`logit_nodes`, with `Node::Rotate` turning the keys' covariance to the query's frame), and the
//! stage's own reads gate on their squared norms beside it.
//!
//! Per layer, a component's gate reads at one of four stages: the attention's input (q, k, v
//! slices), the heads' concatenated outputs (o slices gated on their own read), the MLP's input
//! (c_fc slices) and its activations (down_proj slices gated on their own read). The attention
//! input stage's read `V_A` stacks every q, k and v slice of the stage's components, grouped by
//! component, so `‖V_bᵀx‖` is over all of a component's input-side slices; the gated activations
//! are then split by fixed selections into the q, k and v slices, and each head's q, k and v are
//! the heads' rows of those slices' writes. A write-side slice (o, down_proj) of a component gated
//! at the block's input takes that gate's column (a fixed selection of the gate's columns).
//!
//! Exact frames (an arm's `mixing`, groups of slices of one map each): a group's `n` start slices
//! `(u_j, v_j)` and its extra slices (an arm's `extra`: per site a count of zero slices appended to
//! its factors, listed after the start slices) become reads `V₀ Aᵀ` and writes `U₀ A⁺ + N` with
//! `A` (`C × n`) and `N A = 0` free (`library_mdl::Mix`, which the fit trains), so their sum is the
//! same at every `A` and `N` and a component's slices change within their groups. With any mixing
//! group the explanation is exact by construction: every slice's mean is pinned (`library_mdl` sets
//! the means of the read and write operators to the groups' frames, unmixed slices at their start;
//! their deviations, and so their description, still follow the data), the parts change only within
//! their groups and by their gates, and with slices summing to `M`'s maps every part on at the mean
//! is `M`.
//!
//! The gate ([`Gate`](crate::library_vpd::Gate)). Every Gated node's scale is its stage's operator `{stage}.width` (one
//! entry per component), read as a constant:
//! - `Gate::Hard`, the main arms: the explanation is evaluated with the hard gate `H(z_b)` (the
//!   width holds [`HARD`](crate::library_vpd::HARD)) and trained with its expectation under the posterior, `E_q[H(z_b)]`:
//!   around a pass with a gradient `library_mdl` writes the threshold's posterior deviation `σ_b`
//!   into the width and the threshold's mean into the threshold, so the gate is `Φ(z_b / σ_b)`, the
//!   threshold integrated exactly and the reads and a direction by the pass's weight sample. The
//!   width is no parameter; its operator holds no prior group.
//! - `Gate::Learned`: the width is a trainable parameter (its own prior group, started at the
//!   standard deviation of the component's gate read over the start's fitting tokens), a training
//!   pass gates by `Φ(z_b / w_b)`, and the explanation is scored and exported with the hard gate
//!   `H(z_b)` (`library_mdl::posterior_mean` writes the widths at [`HARD`](crate::library_vpd::HARD)).
//!   The relaxed gate is an aid to the optimization alone: a ramp arm scored by `clamp(z_b / w_b,
//!   0, 1)` scored an explanation other than the one a reader is given.
//! A component counts as active where `z_b > 0`.
//!
//! Prior groups: per slice its read row and its write column (over every head for q, k and v),
//! per direction gate its row, and per stage of a layer its thresholds and its widths (one group
//! each, as a transcoder block's gate biases are). Each head recomputes the attention input stage's gated activations;
//! the heads share the stage's operators.

use crate::{
    artifact::{Argument, Artifact, Callee, Owner},
    explanation_battery::{KINDS, Kind, load_factors},
    library_mdl::{Cells, Explanation, Group, Layer, Share, mean_squares},
    operator_program::{FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Rotary, Rule, Scale, Trace, exact_precision},
    run_check::LayerNodes,
};
use ndarray::{Array1, Array2, Axis, s};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

fn error(e: impl std::fmt::Display) -> String {
    format!("library vpd: {e}")
}

/// A component's gate read, as `vpd_start` writes it.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Read {
    /// `|v_iᵀx|`: slice `index` of `site`, at that site's input (a component's input-side slices).
    Own([usize; 2]),
    /// `gᵀx + c` at `site`'s input (`coefficients`: `g` then `c`).
    Direction { site: usize, coefficients: Vec<f64> },
    /// `‖V_bᵀā‖` at the MLP's input (`site` its c_fc map): the norm of the component's down-slice
    /// reads on the activations `ā` of the layer with every c_fc slice on (a gate on the layer's
    /// own all-on activation, which no linear read of the input gives). Shared (candidates), as a
    /// shared own read: gate `m` on `Σ_b A_mb ‖V_bᵀā‖² − τ_m|τ_m|`.
    Active { site: usize },
    /// `Σ_i c_i(t)² Σ_k u_ik² D_k(t) c²` over the component's q slices (`site` its q map): the
    /// spread of its terms in the attention logits over the keys at query `t` (`c_i` a slice's
    /// read, `u_i` its write, `c` the attention's scale, and per q coordinate `k` of head `h` the
    /// variance `D_k(t)` of the keys' coordinate `k` turned to the query's frame, under head `h`'s
    /// pattern with every q and k slice on; covariances between coordinates left out), against
    /// `τ_b|τ_b|`, as descent's rot arm gates its query blocks. Its stage's own reads gate on their
    /// squared norms `‖V_bᵀx‖² − τ_b|τ_b|` beside it, the same hard gates as `‖V_bᵀx‖ − τ_b`. The
    /// weights `u_ik²` are the start's writes (a fit that moves the writes leaves them there).
    Logits { site: usize },
    /// `Σ_j a_j (ln |g_jᵀx| − θ_j)` over the detectors `j` of `terms` (`(j, a_j)`, a sparse
    /// combination) at `site`'s input stage, against `ln τ_b`: a detector gate. The stage's
    /// detectors ([`DetectorSet`]) are shared directions with thresholds, computed once on the
    /// stage's input (causal: the token's own input), and each part's gate reads a few of them.
    /// With one detector, `a = 1`, `θ = 0` and the detector the part's own rank-one read, it is the
    /// part's own gate `ln |vᵀx| − ln τ`.
    Detectors { site: usize, terms: Vec<(usize, f64)> },
}

/// One stage's shared detectors (`Read::Detectors`): at `layer`'s stage `stage` (0 the attention's
/// input, 2 the MLP's input), per detector its direction `g_j` (over the stage's input) and its
/// threshold `θ_j`, its output `ln max(|g_jᵀx|, t₀) − θ_j`.
#[derive(Clone, Debug, Deserialize)]
struct DetectorSet {
    layer: usize,
    stage: usize,
    directions: Vec<Vec<f64>>,
    thresholds: Vec<f64>,
}

/// One component of a start file.
#[derive(Clone, Debug, Deserialize)]
struct Component {
    read: Read,
    tau: f64,
    /// The gate's starting width `w_b` (`vpd_start`: the standard deviation of its read over the
    /// fitting tokens).
    #[serde(default)]
    width: Option<f64>,
    slices: Vec<[usize; 2]>,
    /// Other components of its stage (indices into the arm's list) whose gates it may move to
    /// during the fit (gate sharing, `library_mdl::Share`); empty keeps it on its own gate.
    #[serde(default)]
    candidates: Vec<usize>,
    /// Beside its own read, its head's largest attention score ([`ScoreRead`]).
    #[serde(default)]
    score: Option<ScoreRead>,
}

/// A component's read of a head's largest pre-softmax score at the token (`Node::MaxScore`, with
/// every q and k slice on, so `M`'s), beside its own read: its gate is
/// `ln ‖V_bᵀx‖ + w s_h(t) − ln τ_b`, its own weight `w` and threshold, at the attention's input
/// (a score gate: what a residual read cannot see, a repeat at an induction position, the head's
/// score shows).
#[derive(Clone, Debug, Deserialize)]
struct ScoreRead {
    head: usize,
    weight: f64,
}

#[derive(Deserialize)]
struct ArmRecord {
    arm: String,
    components: Vec<Component>,
    /// Groups of slices `[site, index]` in exact frames (module note): with any, the explanation is
    /// exact by construction, its slices changing only within their groups.
    #[serde(default)]
    mixing: Vec<Vec<[usize; 2]>>,
    /// Extra slices `[site, count]`: `count` zero slices appended to the site's factors (indices
    /// from the site's slice count on), for the mixing groups' extra slices (module note).
    #[serde(default)]
    extra: Vec<[usize; 2]>,
    /// The stages' shared detectors (`Read::Detectors`).
    #[serde(default)]
    detectors: Vec<DetectorSet>,
    /// Per layer the means its mean parts are made of ([`LayerMeans`]); with them every MLP has
    /// always-on mean parts and its components' parts are deviations from the means.
    #[serde(default)]
    means: Vec<LayerMeans>,
}

/// A layer's means over a calibration text with every part on (`M`'s): of the normed stream into
/// its attention (`attention`, `E[x]`), into its MLP (`mlp`, `E[x]`) and of the MLP's activations
/// (`activation`, `E[act]`); each may be absent (empty). With the MLP's, its c_fc slices read
/// `x − E[x]` and its down slices `act − E[act]`, and two always-on mean parts write
/// `E[p] = W_fc E[x]` into the pre-activations and `W_dn E[act]` into the output. With the
/// attention's, its q, k and v slices read `x − E[x]` and each head's q, k and v get always-on mean
/// parts `W E[x]` (pre-rotation: with its q and k parts off a head attends by position alone, as
/// descent's `DESCENT_MEANQK`), its o slices read the heads' outputs less their mean values
/// (`E[v]`; the pattern's rows sum to one) and an always-on part writes `W_o E[v]`. Every part on
/// is still `M`, and a part off is that part mean-ablated (the activation law's mean is not zero),
/// as descent's mean parts (`DESCENT_MEANPARTS`).
#[derive(Clone, Debug, Deserialize)]
struct LayerMeans {
    layer: usize,
    #[serde(default)]
    attention: Vec<f64>,
    #[serde(default)]
    mlp: Vec<f64>,
    #[serde(default)]
    activation: Vec<f64>,
}

/// Where a slice's read row or write column is in the built explanation, by operator name.
#[derive(Clone, Debug)]
enum Held {
    Read { row: usize },
    Write { column: usize, offset: usize },
}

fn kind_of(site: usize) -> Kind {
    KINDS[site % KINDS.len()]
}

/// The stage of a layer at which a gate reading `site`'s input is computed.
fn stage(site: usize) -> usize {
    match kind_of(site) {
        Kind::Query | Kind::Key | Kind::Value => 0,
        Kind::Output => 1,
        Kind::Up => 2,
        Kind::Down => 3,
    }
}

fn dense(name: &str, rows: Interface, cols: Interface, values: Array2<f64>) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(name, rows, cols, values, precision, Provenance::derived(&[&Provenance::native("vpd")], "VPD slices with intrinsic gates".into())).map_err(error)
}

/// Rows grouped by component: one group per entry of `widths` (all positive).
fn grouped(widths: &[usize]) -> Result<Interface, String> {
    Interface::new(widths.iter().enumerate().map(|(b, &width)| crate::operator_program::Group { width, label: crate::operator_program::Label::new(LabelKind::Unit, b as u32) }).collect()).map_err(error)
}

/// A squared own gate's threshold `−τ|τ|` against `‖V_bᵀx‖²`, the same hard gate as `‖V_bᵀx‖ − τ`;
/// an always-on threshold (`τ < 0`) holds 1, since `−τ|τ|` at an export's `τ = −10⁹` shares an
/// operator's exact lattice with thresholds of order one and rounds them to zero.
fn squared_threshold(tau: f64) -> f64 {
    if tau < 0.0 { 1.0 } else { -tau * tau }
}

/// An unshared own gate's threshold on the log scale of its read norm (toys' guard, cd27062d48): the
/// gate is `ln max(n, t₀) − ln τ` (`Law::Log`, `t₀` its floor), so a part whose reads are zero on a
/// token is off there by `ln τ − ln t₀` (about 87 e-folds at `τ = 1`), and its gate's training noise
/// is relative. A start's `τ ≤ 0` (on everywhere) is one e-fold past every read's log, `1 − ln t₀`.
fn log_threshold(tau: f64) -> f64 {
    if tau > 0.0 { -tau.ln() } else { 1.0 - gam_gpu::tensor::LOG_FLOOR.ln() }
}

/// A learned own gate's width on the log scale, in e-folds: the start's width `w` over its threshold
/// (`d ln n = dn / n` at `n = τ`), one e-fold where `τ ≤ 0`.
fn log_width(width: f64, tau: f64) -> f64 {
    if tau > 0.0 { width / tau } else { 1.0 }
}

fn units(width: usize) -> Result<Interface, String> {
    Interface::uniform(width, 1, LabelKind::Unit, 0).map_err(error)
}

/// A 0/1 selection of `picked` (rows) from `from` columns.
fn selection(picked: &[usize], from: usize) -> Array2<f64> {
    let mut out = Array2::zeros((picked.len(), from));
    for (r, &c) in picked.iter().enumerate() {
        out[[r, c]] = 1.0;
    }
    out
}

fn index_of(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    let found: Vec<usize> = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i).collect();
    match found[..] {
        [i] => Ok(i),
        _ => Err(error(format!("no unique operator {name}"))),
    }
}

/// A gate's law in the explanation (module note).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    /// Evaluated hard, `H(z)`; trained by its expectation under the posterior.
    #[default]
    Hard,
    /// A learned width, trained through `Φ(z / w)` and evaluated by the hard gate `H(z)`
    /// (`DeviceProgram::set_hard`). Trained as `Hard`, the expected gate `Φ(z / σ_τ)` steepens as
    /// the posterior sharpens (IVON sets `σ_τ` from the measured curvature every step, about 10⁻³
    /// after the first), the curvature grows like `1 / σ_τ` and the step stops (vpd4l tiny fit:
    /// the curvature ratio 7e2 → 7e8 within 3 draws, with or without a budget); a width at the
    /// data's scale does not.
    Learned,
}

/// The width a hard gate's stage holds while it is evaluated: `Φ(z / s)` at `s = 10⁻³⁰` (a normal
/// float32; the gate kernels hold `z / s` to ±40) is `H(z)` for every `|z| > 10⁻²⁹`.
pub const HARD: f64 = 1e-30;

/// One arm's explanation of the split native program `native` with its `layers`, from VPD's
/// `decomposition` and the start file `start` (`vpd_start`'s components, arm `arm`), with hard gates.
pub fn explanation(native: &OperatorProgram, layers: &[LayerNodes], decomposition: &Path, start: &Path, arm: &str) -> Result<Explanation, String> {
    explanation_with_gate(native, layers, decomposition, start, arm, Gate::Hard)
}

/// [`explanation`] with the gate law `gate`.
pub fn explanation_with_gate(native: &OperatorProgram, layers: &[LayerNodes], decomposition: &Path, start: &Path, arm: &str, gate: Gate) -> Result<Explanation, String> {
    let mut factors = load_factors(decomposition)?;
    let records: Vec<ArmRecord> = serde_json::from_slice(&std::fs::read(start).map_err(|e| error(format!("{}: {e}", start.display())))?).map_err(error)?;
    let record = records.into_iter().find(|r| r.arm == arm).ok_or_else(|| error(format!("{}: no arm {arm}", start.display())))?;
    let (components, mixing) = (record.components, record.mixing);
    // Each site's start slices, before its extra slices.
    let starts: Vec<usize> = factors.iter().map(|f| f.u.nrows()).collect();
    for &[site, count] in &record.extra {
        let f = factors.get_mut(site).ok_or_else(|| error(format!("extra slices at site {site}, of {}", starts.len())))?;
        f.u.append(Axis(0), Array2::zeros((count, f.u.ncols())).view()).map_err(error)?;
        f.v.append(Axis(1), Array2::zeros((f.v.nrows(), count)).view()).map_err(error)?;
    }
    // Per slice `(site, index)`, where the built explanation holds its read and its write.
    let mut held: BTreeMap<(usize, usize), Vec<(String, Held)>> = BTreeMap::new();
    if factors.len() != KINDS.len() * layers.len() {
        return Err(error(format!("{} sites for {} layers", factors.len(), layers.len())));
    }
    let read_site = |c: &Component| match &c.read {
        Read::Own([site, _]) => *site,
        Read::Direction { site, .. } | Read::Active { site } | Read::Logits { site } | Read::Detectors { site, .. } => *site,
    };
    let detector_sets = record.detectors;
    let layer_means = record.means;
    // The detectors of `layer`'s stage `stage`, where a stage gates on them.
    let detector_set = |layer: usize, stage: usize| detector_sets.iter().find(|d| d.layer == layer && d.stage == stage);
    let mut artifact = Artifact::native(native)?;
    let mut shares = Vec::new();
    // Per layer and stage, the components gated there (each component's home, its gate's stage).
    // A component's slices sit at its home, at the stage its home's block carries (the o slices
    // after the attention's input, the down slices after the MLP's input), or at any later stage of
    // any later layer: a block of any rank across maps and layers under its one gate. A slice in
    // another block is carried there (`cross`): that block's rule recomputes the gate from the home's
    // input, which it takes as an argument, so a carried component's gate is an unshared direction
    // gate at the attention's or the MLP's input, whose input `M`'s program holds.
    let home_of = |site: usize| (site / KINDS.len(), stage(site));
    let mut at: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    let mut cross: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (b, c) in components.iter().enumerate() {
        let home = home_of(read_site(c));
        let mut into = std::collections::BTreeSet::new();
        for &[s, _] in &c.slices {
            let place = home_of(s);
            if place < home {
                return Err(error(format!("component {b}: a slice before its gate's stage")));
            }
            if place != home && !(place.0 == home.0 && matches!((home.1, place.1), (0, 1) | (2, 3))) {
                into.insert(place);
            }
        }
        if !into.is_empty() && (!matches!(c.read, Read::Direction { .. }) || !c.candidates.is_empty() || !matches!(home.1, 0 | 2)) {
            return Err(error(format!("component {b}: slices carried into another block need an unshared direction gate at the attention's or the MLP's input")));
        }
        for place in into {
            cross.entry(place).or_default().push(b);
        }
        at.entry(home).or_default().push(b);
    }
    // The components carried into the stage `place` (`cross`), by home in home order, each home's
    // in its order there: per home its components and their positions among the home's.
    let carried_into = |place: (usize, usize)| -> Vec<((usize, usize), Vec<usize>, Vec<usize>)> {
        let mut by_home: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
        for &b in cross.get(&place).into_iter().flatten() {
            let home = home_of(read_site(&components[b]));
            let position = at.get(&home).and_then(|comps| comps.iter().position(|c| *c == b)).unwrap_or(0);
            by_home.entry(home).or_default().push((position, b));
        }
        by_home
            .into_iter()
            .map(|(home, mut list)| {
                list.sort_unstable();
                (home, list.iter().map(|p| p.1).collect(), list.iter().map(|p| p.0).collect())
            })
            .collect()
    };
    let home_prefix = |(l, s): (usize, usize)| if s == 0 { format!("library.l{l}.attn") } else { format!("library.l{l}.mlp.fc") };
    let home_input = |(l, s): (usize, usize)| if s == 0 { layers[l].normed_stream } else { layers[l].normed };
    // The distinct inputs of `carried`'s homes not among `inputs`, in order.
    let extra_inputs = |carried: &[((usize, usize), Vec<usize>, Vec<usize>)], inputs: &[usize]| -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        for (home, _, _) in carried {
            let node = home_input(*home);
            if !inputs.contains(&node) && !out.contains(&node) {
                out.push(node);
            }
        }
        out
    };
    // In a rule whose inputs are the native nodes `inputs` (parameter `i` the `i`-th), the carried
    // components' gate pre-activations and widths: per home its direction, threshold and width on
    // its input, selected by `{consumer}.from.{home}` (pushed to `operators`, base `base`, unless a
    // rule built before holds it).
    let cross_parts = |artifact: &Artifact, nodes: &mut Vec<Node>, (operators, base): (&mut Vec<Operator>, usize), consumer: &str, carried: &[((usize, usize), Vec<usize>, Vec<usize>)], inputs: &[usize]| -> Result<(Vec<usize>, Vec<usize>), String> {
        let (mut zs, mut ss) = (Vec::new(), Vec::new());
        for (home, comps, positions) in carried {
            let prefix = home_prefix(*home);
            let op = |part: &str| index_of(&artifact.program, &format!("{prefix}.{part}"));
            let name = format!("{consumer}.from.{prefix}");
            let select = match (index_of(&artifact.program, &name), operators.iter().position(|o| o.name == name)) {
                (Ok(i), _) => i,
                (Err(_), Some(i)) => base + i,
                (Err(_), None) => {
                    let count = at.get(home).map_or(0, Vec::len);
                    operators.push(dense(&name, units(comps.len())?, units(count)?, selection(positions, count))?);
                    base + operators.len() - 1
                }
            };
            let param = inputs.iter().position(|n| *n == home_input(*home)).ok_or_else(|| error(format!("{consumer}: no input of its carried gates' home {prefix}")))?;
            nodes.push(Node::Affine { terms: vec![(param, op("direction")?)], bias: Some(op("threshold")?) });
            nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, select)], bias: None });
            zs.push(nodes.len() - 1);
            nodes.push(Node::Constant { operator: op("width")? });
            nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, select)], bias: None });
            ss.push(nodes.len() - 1);
        }
        Ok((zs, ss))
    };
    // Per stage its gate reads' kind: direction (true) or own (false), refused where mixed.
    // A detector gate is a direction gate on the stage's detectors (`Read::Detectors`).
    let direction_of = |comps: &[usize]| -> Result<bool, String> {
        let detectors = comps.iter().filter(|&&b| matches!(components[b].read, Read::Detectors { .. })).count();
        if detectors != 0 && detectors != comps.len() {
            return Err(error(format!("a stage with {detectors} detector gate reads among {} (a stage gates one way)", comps.len())));
        }
        let directions = comps.iter().filter(|&&b| matches!(components[b].read, Read::Direction { .. } | Read::Detectors { .. })).count();
        if directions != 0 && directions != comps.len() {
            return Err(error(format!("a stage with {directions} direction and {} own gate reads (a stage gates one way)", comps.len() - directions)));
        }
        Ok(directions != 0)
    };
    // Per stage whether its gates read the all-on activations (`Read::Active`: every component of
    // an MLP input stage, each carrying down slices, or none).
    let active_of = |comps: &[usize]| -> Result<bool, String> {
        let active = comps.iter().filter(|&&b| matches!(components[b].read, Read::Active { .. })).count();
        if active != 0 && active != comps.len() {
            return Err(error(format!("a stage with {active} all-on activation reads among {} (a stage gates one way)", comps.len())));
        }
        Ok(active != 0)
    };
    // Per stage whether its gates read the attention logits' spread (`Read::Logits`): q
    // components at the attention's input, unshared, beside own reads on squared norms.
    let logits_of = |comps: &[usize]| comps.iter().any(|&b| matches!(components[b].read, Read::Logits { .. }));
    for (&(l, stage_index), comps) in &at {
        direction_of(comps)?;
        let detecting = comps.iter().any(|&b| matches!(components[b].read, Read::Detectors { .. }));
        match (detecting, detector_set(l, stage_index)) {
            (true, None) => return Err(error(format!("layer {l} stage {stage_index}: detector gate reads without the stage's detectors"))),
            (false, Some(_)) => return Err(error(format!("layer {l} stage {stage_index}: detectors with no detector gate reading them"))),
            (true, Some(set)) => {
                let k = set.directions.len();
                if k == 0 || set.thresholds.len() != k || !(stage_index == 0 || stage_index == 2) {
                    return Err(error(format!("layer {l} stage {stage_index}: detectors at the attention's or the MLP's input, each with one threshold")));
                }
                for &b in comps {
                    if let Read::Detectors { terms, .. } = &components[b].read
                        && (terms.is_empty() || terms.iter().any(|(j, a)| *j >= k || *a == 0.0))
                    {
                        return Err(error(format!("component {b}: a detector gate reads some of its stage's {k} detectors, each with a nonzero weight")));
                    }
                }
            }
            (false, None) => {}
        }
        if active_of(comps)? && stage_index != 2 {
            return Err(error("all-on activation reads gate components at the MLP's input only"));
        }
        if logits_of(comps) {
            if stage_index != 0 || comps.iter().any(|&b| !components[b].candidates.is_empty()) {
                return Err(error(format!("layer {l}: logit reads gate unshared components at the attention's input only")));
            }
            for &b in comps {
                if let Read::Logits { site } = components[b].read
                    && (kind_of(site) != Kind::Query || site / KINDS.len() != l || components[b].slices.iter().any(|[s, _]| *s != site))
                {
                    return Err(error(format!("component {b}: a logit read gates q slices of its own layer's q map alone")));
                }
            }
        }
    }
    let interfaces = native.interfaces().map_err(error)?;
    let node_interface = |node: usize| -> Result<Interface, String> { interfaces.get(node).cloned().ok_or_else(|| error(format!("no node {node}"))) };
    // A stage is shared where any of its components lists candidates (gate sharing).
    let is_shared = |comps: &[usize]| comps.iter().any(|&b| !components[b].candidates.is_empty());
    // Its record: per component its candidate gates, its own first, as positions in the stage.
    let share_of = |prefix: &str, comps: &[usize]| -> Result<Share, String> {
        let position: BTreeMap<usize, usize> = comps.iter().enumerate().map(|(i, b)| (*b, i)).collect();
        let candidates = comps
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let mut list = vec![i];
                for c in &components[b].candidates {
                    let at = *position.get(c).ok_or_else(|| error(format!("component {b}: candidate {c} is not in its stage")))?;
                    if !list.contains(&at) {
                        list.push(at);
                    }
                }
                Ok(list)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Share { operator: format!("{prefix}.assign"), candidates })
    };
    for (l, layer) in layers.iter().enumerate() {
        let name = format!("library.l{l}");
        let site = |kind: Kind| KINDS.len() * l + KINDS.iter().position(|k| *k == kind).unwrap_or(0);
        let slices_on = |b: usize, s: usize| -> Vec<usize> { components[b].slices.iter().filter(|[t, _]| *t == s).map(|[_, i]| *i).collect() };
        // ---------------------------------------------------------------- the attention's input stage
        let a_comps = at.get(&(l, 0)).cloned().unwrap_or_default();
        let attn_shared = is_shared(&a_comps);
        let a_direction = direction_of(&a_comps)?;
        if attn_shared {
            shares.push(share_of(&format!("{name}.attn"), &a_comps)?);
        }
        let (q, k, v) = (site(Kind::Query), site(Kind::Key), site(Kind::Value));
        let a_cross = carried_into((l, 0));
        if !a_cross.is_empty() && attn_shared {
            return Err(error(format!("layer {l}: components carried into a shared attention input stage")));
        }
        // The stacked read: per component (the carried ones first) its q, k, v slices in that
        // order; each slice's row.
        let mut read_rows: Vec<(usize, usize)> = Vec::new();
        let mut widths = Vec::new();
        let a_carried: usize = a_cross.iter().map(|(_, comps, _)| comps.len()).sum();
        for &b in a_cross.iter().flat_map(|(_, comps, _)| comps).chain(&a_comps) {
            let mut w = 0;
            for s in [q, k, v] {
                for i in slices_on(b, s) {
                    read_rows.push((s, i));
                    w += 1;
                }
            }
            if w == 0 {
                return Err(error(format!("component {b}: gated at the attention's input with no q, k or v slice")));
            }
            widths.push(w);
        }
        if a_comps.is_empty() {
            return Err(error(format!("layer {l}: no component at the attention's input")));
        }
        let x = layer.normed_stream;
        if layer.reads.is_empty() {
            return Err(error(format!("layer {l}: no heads")));
        }
        // Each head's q, k and v are one map of the normed stream (no head norm).
        for &read in &layer.reads {
            let Node::Attend { query, key, value, .. } = &native.nodes[read] else { return Err(error(format!("layer {l}: a head read is not an attention node"))) };
            for node in [*query, *key, *value] {
                if !matches!(&native.nodes[node], Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == x) {
                    return Err(error(format!("layer {l}: a head's projection is not one map of the normed stream")));
                }
            }
        }
        let x_interface = node_interface(x)?;
        let d = x_interface.width();
        let stacked = grouped(&widths)?;
        let r_a = read_rows.len();
        let v_a = Array2::from_shape_fn((r_a, d), |(r, j)| factors[read_rows[r].0].v[[j, read_rows[r].1]]);
        // `share`: the components share the stage's gates (an assignment, own gates on the squared
        // norm of their members' reads).
        // `detectors`: the stage's detectors, where its gates read them (`Read::Detectors`): the
        // gates' direction rows are the combinations over the detectors and the detectors'
        // operators follow the stage's others.
        // `heads`: a score stage's heads (`ScoreRead`): its gate rows read its own groups' log norms
        // and every head's largest score (`score_features`), each row its own column and its head's.
        let gate_ops = |prefix: &str, comps: &[usize], cols: &Interface, share: bool, detectors: Option<&DetectorSet>, heads: Option<usize>| -> Result<Vec<Operator>, String> {
            let count = comps.len();
            let direction = direction_of(comps)?;
            // A stage with logit reads gates on squared reads (`Read::Logits`).
            let squared = logits_of(comps);
            let mut ops = Vec::new();
            match direction {
                false if heads.is_some() => {
                    let h = heads.unwrap_or(0);
                    let g = Array2::from_shape_fn((count, count + h), |(b, j)| match &components[comps[b]].score {
                        _ if j == b => 1.0,
                        Some(ScoreRead { head, weight }) if j == count + head => *weight,
                        _ => 0.0,
                    });
                    ops.push(dense(&format!("{prefix}.direction"), units(count)?, score_features(count, h)?, g)?);
                    ops.push(dense(&format!("{prefix}.threshold"), units(count)?, Interface::constant(), Array2::from_shape_fn((count, 1), |(b, _)| log_threshold(components[comps[b]].tau)))?);
                }
                false if share => {
                    // Shared own gates read the squared norm of their components' reads: gate m's
                    // pre-activation Σ_b A_mb ‖V_bᵀx‖² − τ_m|τ_m|, which at a 0/1 assignment is on
                    // exactly where the norm of its members' reads exceeds τ_m.
                    ops.push(dense(&format!("{prefix}.assign"), units(count)?, units(count)?, Array2::eye(count))?);
                    ops.push(dense(&format!("{prefix}.threshold"), units(count)?, Interface::constant(), Array2::from_shape_fn((count, 1), |(b, _)| squared_threshold(components[comps[b]].tau)))?);
                }
                false => {
                    let tau = |b: usize| components[comps[b]].tau;
                    ops.push(Operator::identity(format!("{prefix}.gate_identity"), units(count)?));
                    ops.push(dense(&format!("{prefix}.threshold"), units(count)?, Interface::constant(), Array2::from_shape_fn((count, 1), |(b, _)| if squared { squared_threshold(tau(b)) } else { log_threshold(tau(b)) }))?);
                }
                true if detectors.is_some() => {
                    let k = detectors.map_or(0, |d| d.directions.len());
                    let g = Array2::from_shape_fn((count, k), |(b, j)| match &components[comps[b]].read {
                        Read::Detectors { terms, .. } => terms.iter().filter(|(i, _)| *i == j).map(|(_, a)| a).sum(),
                        _ => 0.0,
                    });
                    ops.push(dense(&format!("{prefix}.direction"), units(count)?, units(k)?, g)?);
                    ops.push(dense(&format!("{prefix}.threshold"), units(count)?, Interface::constant(), Array2::from_shape_fn((count, 1), |(b, _)| log_threshold(components[comps[b]].tau)))?);
                }
                true => {
                    let w = cols.width();
                    let g = Array2::from_shape_fn((count, w), |(b, j)| match &components[comps[b]].read {
                        Read::Direction { coefficients, .. } => coefficients[j],
                        Read::Own(_) | Read::Active { .. } | Read::Logits { .. } | Read::Detectors { .. } => 0.0,
                    });
                    let c = Array2::from_shape_fn((count, 1), |(b, _)| match &components[comps[b]].read {
                        Read::Direction { coefficients, .. } => coefficients[w] - components[comps[b]].tau,
                        Read::Own(_) | Read::Active { .. } | Read::Logits { .. } | Read::Detectors { .. } => 0.0,
                    });
                    ops.push(dense(&format!("{prefix}.direction"), units(count)?, cols.clone(), g)?);
                    ops.push(dense(&format!("{prefix}.threshold"), units(count)?, Interface::constant(), c)?);
                }
            }
            let widths = match gate {
                Gate::Hard => vec![HARD; count],
                Gate::Learned => comps.iter().map(|&b| components[b].width.filter(|w| w.is_finite() && *w > 0.0).ok_or_else(|| error(format!("component {b}: no positive gate width in the start file (rerun mpd_battery_2951 start)")))).collect::<Result<Vec<f64>, String>>()?,
            };
            // A shared own gate's width on the squared norm has the norm's slope at the threshold:
            // d‖·‖²/d‖·‖ = 2τ there (2w for a threshold below one width); an unshared own gate's
            // is on the log scale ([`log_width`]).
            let width = |b: usize| match (share || squared, direction, gate) {
                (_, _, Gate::Hard) | (_, true, _) => widths[b],
                (true, false, Gate::Learned) => 2.0 * widths[b] * components[comps[b]].tau.max(widths[b]),
                (false, false, Gate::Learned) => log_width(widths[b], components[comps[b]].tau),
            };
            ops.push(dense(&format!("{prefix}.width"), units(count)?, Interface::constant(), Array2::from_shape_fn((count, 1), |(b, _)| width(b)))?);
            if share && direction {
                ops.push(dense(&format!("{prefix}.assign"), units(count)?, units(count)?, Array2::eye(count))?);
            }
            if let Some(h) = heads {
                ops.push(Operator::identity(format!("{prefix}.score_heads"), score_heads(h)?));
            }
            if let Some(set) = detectors {
                let (k, d) = (set.directions.len(), cols.width());
                if set.directions.iter().any(|g| g.len() != d) {
                    return Err(error(format!("{prefix}: detector directions of another width than the stage's input {d}")));
                }
                ops.push(dense(&format!("{prefix}.detectors"), units(k)?, cols.clone(), Array2::from_shape_fn((k, d), |(j, i)| set.directions[j][i]))?);
                ops.push(Operator::identity(format!("{prefix}.detector_identity"), units(k)?));
                ops.push(dense(&format!("{prefix}.detector_thresholds"), units(k)?, Interface::constant(), Array2::from_shape_fn((k, 1), |(j, _)| -set.thresholds[j]))?);
            }
            Ok(ops)
        };
        // A stage's detector operators (`Read::Detectors`), with their count: by name among a rule's
        // own operators (from `base`), else in the artifact.
        let detector_ops = |artifact: &Artifact, operators: &[Operator], base: usize, prefix: &str| -> Option<(usize, usize, usize, usize)> {
            let at = |part: &str| {
                let n = format!("{prefix}.{part}");
                operators.iter().position(|o| o.name == n).map(|i| (base + i, operators[i].rows.width())).or_else(|| index_of(&artifact.program, &n).ok().map(|i| (i, artifact.program.operators[i].rows.width())))
            };
            let (d, k) = at("detectors")?;
            Some((d, at("detector_identity")?.0, at("detector_thresholds")?.0, k))
        };
        // The attention input stage's nodes from `input` (node 0 of a rule's nodes so far): the
        // stacked read, the gate, the gate's width, the gated activations; returns (gated node,
        // gate node, width node).
        // With `extra` (the carried components' gate parts), the gated reads, gated by the carried
        // gates then the stage's own; without, its own gates alone. `own` selects the own
        // components' read norms where carried rows come first in the read; an unshared own gate
        // reads their logs (`groups` of them, [`log_threshold`]).
        // A detector stage's gate reads its detectors' outputs `ln max(|Gx|, t₀) − θ` (`detectors`:
        // their operators and count, [`Read::Detectors`]) in place of `input`.
        // `reads`: the node the stage's slices read (`input` less its mean with mean parts,
        // [`LayerMeans`]); direction and detector gates read `input` itself.
        let stage_nodes = |nodes: &mut Vec<Node>, (input, reads): (usize, usize), (read, gate_a, gate_b, soft): (usize, usize, usize, usize), assign: Option<usize>, direction: bool, own: Option<usize>, extra: Option<(Vec<usize>, Vec<usize>)>, groups: usize, detectors: Option<(usize, usize, usize, usize)>| -> (Option<usize>, usize, usize) {
            let gate_input = match detectors {
                Some((g, identity, thresholds, k)) => {
                    nodes.push(Node::Affine { terms: vec![(input, g)], bias: None });
                    nodes.push(Node::GroupNorm { input: nodes.len() - 1 });
                    nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![Law::Log; k] });
                    nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, identity)], bias: Some(thresholds) });
                    nodes.len() - 1
                }
                None => input,
            };
            // A shared stage (`assign`, gates × components): the gates' pre-activations and widths,
            // then each component's own, z_b = Σ_m A_mb z_m and w_b = Σ_m A_mb w_m (a 0/1
            // assignment gives each component its gate's), before the reads.
            if let Some(assign) = assign {
                let (a, z) = if direction {
                    nodes.push(Node::Affine { terms: vec![(gate_input, gate_a)], bias: Some(gate_b) });
                    (None, nodes.len() - 1)
                } else {
                    nodes.push(Node::Affine { terms: vec![(reads, read)], bias: None });
                    let a = nodes.len() - 1;
                    nodes.push(Node::GroupNorm { input: a });
                    nodes.push(Node::Hadamard { left: a + 1, right: a + 1 });
                    nodes.push(Node::Affine { terms: vec![(a + 2, assign)], bias: Some(gate_b) });
                    (Some(a), nodes.len() - 1)
                };
                nodes.push(Node::Constant { operator: soft });
                let w = nodes.len() - 1;
                nodes.push(Node::Transposed { input: z, operator: assign });
                nodes.push(Node::Transposed { input: w, operator: assign });
                let (zb, wb) = (nodes.len() - 2, nodes.len() - 1);
                let a = a.unwrap_or_else(|| {
                    nodes.push(Node::Affine { terms: vec![(reads, read)], bias: None });
                    nodes.len() - 1
                });
                let gated = extra.map(|_| {
                    nodes.push(Node::Gated { value: a, gate: zb, scale: Some(wb) });
                    nodes.len() - 1
                });
                return (gated, zb, wb);
            }
            // A direction gate and its width come before the reads they gate, so each row reads
            // only its components on (`DeviceProgram`'s gated reads); an own gate reads them.
            let (a, z, s) = if direction {
                nodes.push(Node::Affine { terms: vec![(gate_input, gate_a)], bias: Some(gate_b) });
                nodes.push(Node::Constant { operator: soft });
                nodes.push(Node::Affine { terms: vec![(reads, read)], bias: None });
                (nodes.len() - 1, nodes.len() - 3, nodes.len() - 2)
            } else {
                nodes.push(Node::Affine { terms: vec![(reads, read)], bias: None });
                let a = nodes.len() - 1;
                nodes.push(Node::GroupNorm { input: a });
                nodes.push(Node::Pointwise { input: a + 1, laws: vec![Law::Log; groups] });
                if let Some(select) = own {
                    nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, select)], bias: None });
                }
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, gate_a)], bias: Some(gate_b) });
                nodes.push(Node::Constant { operator: soft });
                (a, nodes.len() - 2, nodes.len() - 1)
            };
            let gated = extra.map(|(mut zs, mut ss)| {
                zs.push(z);
                ss.push(s);
                let (zj, sj) = joined(nodes, zs, ss);
                nodes.push(Node::Gated { value: a, gate: zj, scale: Some(sj) });
                nodes.len() - 1
            });
            (gated, z, s)
        };
        // The attention input stage's own read norms where carried rows come first.
        let a_own = a_carried > 0 && !a_direction;
        let selections: Vec<(usize, Vec<usize>)> = [q, k, v].iter().map(|&s| (s, (0..r_a).filter(|&r| read_rows[r].0 == s).collect())).collect();
        // A logit-read stage (`Read::Logits`): every head's rule computes the spread of every
        // head's logits, so the first head's rule makes every head's q and k writes and the
        // spread's constants, and every rule reads them. Per stacked row whether a logit read's.
        let a_logits = logits_of(&a_comps);
        // A score stage (`ScoreRead`): own reads, unshared, uncarried, beside no logit read.
        let a_scored = a_comps.iter().any(|&b| components[b].score.is_some());
        if a_scored
            && (a_logits || a_carried > 0 || attn_shared || direction_of(&a_comps)? || a_comps.iter().any(|&b| components[b].score.as_ref().is_some_and(|r| r.head >= layer.reads.len())))
        {
            return Err(error(format!("layer {l}: score reads gate unshared own-read components at the attention's input, with no carried component, logit read or head past {}", layer.reads.len())));
        }
        if a_logits && (a_carried > 0 || a_comps.iter().any(|&b| !slices_on(b, site(Kind::Output)).is_empty())) {
            return Err(error(format!("layer {l}: a logit-read stage with carried components or o slices")));
        }
        // The attention's mean parts (`LayerMeans`): `E[x]` at its input, and per head `M`'s q, k and
        // v of it (pre-rotation).
        let attn_means: Option<(Array1<f64>, Vec<[Array1<f64>; 3]>)> = match layer_means.iter().find(|m| m.layer == l && !m.attention.is_empty()) {
            Some(m) => {
                if m.attention.len() != d || a_logits {
                    return Err(error(format!("layer {l}: attention means of {} for an input of {d}, or beside logit reads", m.attention.len())));
                }
                let ex = Array1::from(m.attention.clone());
                let map = |node: usize| -> Result<Array1<f64>, String> {
                    match &native.nodes[node] {
                        Node::Affine { terms, bias: None } if terms.len() == 1 => Ok(native.operators[terms[0].1].matrix().dot(&ex)),
                        other => Err(error(format!("layer {l}: a head's projection is {other:?}"))),
                    }
                };
                let heads = layer
                    .reads
                    .iter()
                    .map(|&read| match &native.nodes[read] {
                        Node::Attend { query, key, value, .. } => Ok([map(*query)?, map(*key)?, map(*value)?]),
                        _ => Err(error(format!("layer {l}: a head read is not an attention node"))),
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Some((ex, heads))
            }
            None => None,
        };
        let logit_rows: Vec<bool> = a_comps.iter().flat_map(|&b| [q, k, v].map(|s| (slices_on(b, s).len(), matches!(components[b].read, Read::Logits { .. })))).flat_map(|(n, logit)| vec![logit; n]).collect();
        for (h, &read) in layer.reads.iter().enumerate() {
            let Node::Attend { query, key, value, scale, rotary, causal } = native.nodes[read].clone() else {
                return Err(error(format!("layer {l} head {h}: the read is not an attention node")));
            };
            let head_rows = |node: usize| -> Result<Interface, String> { node_interface(node) };
            let hd = head_rows(query)?.width();
            let base = artifact.program.operators.len();
            let mut operators = Vec::new();
            let shared = |artifact: &Artifact, part: &str| index_of(&artifact.program, &format!("{name}.attn.{part}"));
            let (read_op, gate_a, gate_b, soft) = if h == 0 {
                operators.push(dense(&format!("{name}.attn.read"), stacked.clone(), x_interface.clone(), v_a.clone())?);
                for (row, slice) in read_rows.iter().enumerate() {
                    held.entry(*slice).or_default().push((format!("{name}.attn.read"), Held::Read { row }));
                }
                let mut gates = gate_ops(&format!("{name}.attn"), &a_comps, &x_interface, attn_shared, detector_set(l, 0), a_scored.then_some(layer.reads.len()))?;
                operators.append(&mut gates);
                for (s, picked) in &selections {
                    if !picked.is_empty() {
                        operators.push(dense(&format!("{name}.attn.select{}", s % KINDS.len()), units(picked.len())?, stacked.clone(), selection(picked, r_a))?);
                    }
                }
                if a_own {
                    let groups = a_carried + a_comps.len();
                    operators.push(dense(&format!("{name}.attn.select_own"), units(a_comps.len())?, units(groups)?, selection(&(a_carried..groups).collect::<Vec<_>>(), groups))?);
                }
                if a_logits || a_scored {
                    for (h2, &read2) in layer.reads.iter().enumerate() {
                        let Node::Attend { query: q2, key: k2, scale: c2, rotary: r2, causal: m2, .. } = native.nodes[read2].clone() else {
                            return Err(error(format!("layer {l} head {h2}: the read is not an attention node")));
                        };
                        if (c2, r2, m2) != (scale, rotary, causal) || rotary.is_some_and(|r| r.dims as usize != hd) {
                            return Err(error(format!("layer {l}: a logit read needs one scale, mask and whole-head rotation for every head")));
                        }
                        for (j, node) in [(0, q2), (1, k2)] {
                            let (s, picked) = &selections[j];
                            let w_name = format!("{name}.h{h2}.{}", ["q", "k"][j]);
                            let rows = head_rows(node)?;
                            let u = Array2::from_shape_fn((rows.width(), picked.len()), |(r, c)| factors[*s].u[[read_rows[picked[c]].1, h2 * hd + r]]);
                            for (column, &r) in picked.iter().enumerate() {
                                held.entry(read_rows[r]).or_default().push((w_name.clone(), Held::Write { column, offset: h2 * hd }));
                            }
                            operators.push(dense(&w_name, rows, units(picked.len())?, u)?);
                        }
                    }
                }
                if let Some((ex, heads)) = &attn_means {
                    operators.push(Operator::identity(format!("{name}.attn.center"), x_interface.clone()));
                    operators.push(dense(&format!("{name}.attn.center_bias"), x_interface.clone(), Interface::constant(), (-ex).insert_axis(Axis(1)))?);
                    // A score stage's rules read every head's q and k: its first makes every head's q and k
                    // means.
                    for (h2, (means, &read2)) in heads.iter().zip(&layer.reads).enumerate().filter(|_| a_scored) {
                        let Node::Attend { query: q2, key: k2, .. } = native.nodes[read2].clone() else {
                            return Err(error(format!("layer {l} head {h2}: the read is not an attention node")));
                        };
                        for (j, node) in [q2, k2].into_iter().enumerate() {
                            operators.push(dense(&format!("{name}.h{h2}.{}_mean", ["q", "k", "v"][j]), head_rows(node)?, Interface::constant(), means[j].clone().insert_axis(Axis(1)))?);
                        }
                    }
                }
                if a_logits {
                    let u = |r: usize, c: usize| factors[q].u[[read_rows[r].1, c]];
                    operators.append(&mut spread_ops(&format!("{name}.attn"), (&head_rows(key)?, layer.reads.len()), (rotary, scale.value()), (&stacked, &widths, &logit_rows), &u)?);
                }
                (base, base + 1, base + 2, base + 3)
            } else {
                let gate_names = match (a_direction || a_scored, attn_shared) {
                    (true, _) => ("direction", "threshold"),
                    (false, true) => ("assign", "threshold"),
                    (false, false) => ("gate_identity", "threshold"),
                };
                (shared(&artifact, "read")?, shared(&artifact, gate_names.0)?, shared(&artifact, gate_names.1)?, shared(&artifact, "width")?)
            };
            // Each head's rule makes its own v mean, and outside a score stage its q and k means.
            if let Some((_, heads)) = &attn_means {
                for (j, node) in [query, key, value].into_iter().enumerate().filter(|(j, _)| !a_scored || *j == 2) {
                    operators.push(dense(&format!("{name}.h{h}.{}_mean", ["q", "k", "v"][j]), head_rows(node)?, Interface::constant(), heads[h][j].clone().insert_axis(Axis(1)))?);
                }
            }
            let assign = match (attn_shared, h) {
                (false, _) => None,
                (true, 0) => Some(if a_direction { base + 4 } else { base + 1 }),
                (true, _) => Some(shared(&artifact, "assign")?),
            };
            let mut inputs = vec![x];
            inputs.extend(extra_inputs(&a_cross, &inputs));
            let mut nodes: Vec<Node> = (0..inputs.len()).map(|index| Node::Param { index }).collect();
            let parts = cross_parts(&artifact, &mut nodes, (&mut operators, base), &format!("{name}.attn"), &a_cross, &inputs)?;
            let own = match (a_own, h) {
                (false, _) => None,
                (true, 0) => Some(base + operators.iter().position(|o| o.name == format!("{name}.attn.select_own")).ok_or("the own selection")?),
                (true, _) => Some(shared(&artifact, "select_own")?),
            };
            // An operator of the stage by name, the first head's rule's or the program's.
            let find = |artifact: &Artifact, operators: &[Operator], part: &str| -> Result<usize, String> {
                match operators.iter().position(|o| o.name == part) {
                    Some(i) => Ok(base + i),
                    None => index_of(&artifact.program, part),
                }
            };
            // With mean parts, the slices read `x − E[x]` (`LayerMeans`), and each head's q, k and v
            // add their means.
            let reads_from = match &attn_means {
                Some(_) => {
                    nodes.push(Node::Affine { terms: vec![(0, find(&artifact, &operators, &format!("{name}.attn.center"))?)], bias: Some(find(&artifact, &operators, &format!("{name}.attn.center_bias"))?) });
                    nodes.len() - 1
                }
                None => 0,
            };
            // Per head its q, k and v mean parts.
            let mean_of = |h2: usize, j: usize| find(&artifact, &operators, &format!("{name}.h{h2}.{}_mean", ["q", "k", "v"][j]));
            // This head's means, and a score stage's every head's q and k means.
            let own_means: Option<[usize; 3]> = match &attn_means {
                Some(_) => Some([mean_of(h, 0)?, mean_of(h, 1)?, mean_of(h, 2)?]),
                None => None,
            };
            let score_means: Option<Vec<(usize, usize)>> = match (&attn_means, a_scored) {
                (Some(_), true) => Some((0..layer.reads.len()).map(|h2| Ok((mean_of(h2, 0)?, mean_of(h2, 1)?))).collect::<Result<Vec<_>, String>>()?),
                _ => None,
            };
            let (gated, _, _) = if a_logits {
                let writes: Vec<(usize, usize)> = (0..layer.reads.len())
                    .map(|h2| Ok((find(&artifact, &operators, &format!("{name}.h{h2}.q"))?, find(&artifact, &operators, &format!("{name}.h{h2}.k"))?)))
                    .collect::<Result<_, String>>()?;
                let part = |p: &str| find(&artifact, &operators, &format!("{name}.attn.{p}"));
                let groups = head_rows(key)?.group_count();
                let (a, sum) = logit_nodes(&mut nodes, (0, read_op, part("select0")?, part("select1")?), &writes, (scale, rotary, causal, groups), &part)?;
                nodes.push(Node::Affine { terms: vec![(sum, gate_a)], bias: Some(gate_b) });
                let z = nodes.len() - 1;
                nodes.push(Node::Constant { operator: soft });
                let w = nodes.len() - 1;
                nodes.push(Node::Gated { value: a, gate: z, scale: Some(w) });
                (Some(nodes.len() - 1), z, w)
            } else if a_scored {
                let writes: Vec<(usize, usize)> = (0..layer.reads.len())
                    .map(|h2| Ok((find(&artifact, &operators, &format!("{name}.h{h2}.q"))?, find(&artifact, &operators, &format!("{name}.h{h2}.k"))?)))
                    .collect::<Result<_, String>>()?;
                let part = |p: &str| find(&artifact, &operators, &format!("{name}.attn.{p}"));
                let (a, z) = score_gate_nodes(&mut nodes, (reads_from, read_op, part("select0")?, part("select1")?, part("score_heads")?), (&writes, score_means.as_deref()), (scale, rotary, causal, widths.len()), (gate_a, gate_b));
                nodes.push(Node::Constant { operator: soft });
                let w = nodes.len() - 1;
                nodes.push(Node::Gated { value: a, gate: z, scale: Some(w) });
                (Some(nodes.len() - 1), z, w)
            } else {
                stage_nodes(&mut nodes, (0, reads_from), (read_op, gate_a, gate_b, soft), assign, a_direction, own, Some(parts), widths.len(), detector_ops(&artifact, &operators, base, &format!("{name}.attn")))
            };
            let gated = gated.ok_or("the gated reads")?;
            let mut projections = Vec::new();
            for (j, (s, picked)) in selections.iter().enumerate() {
                let rows = head_rows([query, key, value][j])?;
                let w_name = format!("{name}.h{h}.{}", ["q", "k", "v"][j]);
                if picked.is_empty() {
                    return Err(error(format!("layer {l}: no {} slice", ["q", "k", "v"][j])));
                }
                let select = if h == 0 {
                    base + operators.iter().position(|o| o.name == format!("{name}.attn.select{}", s % KINDS.len())).ok_or("selection")?
                } else {
                    shared(&artifact, &format!("select{}", s % KINDS.len()))?
                };
                let write = if (a_logits || a_scored) && j < 2 {
                    find(&artifact, &operators, &w_name)?
                } else {
                    let u = Array2::from_shape_fn((rows.width(), picked.len()), |(r, c)| factors[*s].u[[read_rows[picked[c]].1, h * hd + r]]);
                    for (column, &r) in picked.iter().enumerate() {
                        held.entry(read_rows[r]).or_default().push((w_name.clone(), Held::Write { column, offset: h * hd }));
                    }
                    operators.push(dense(&w_name, rows, units(picked.len())?, u)?);
                    base + operators.len() - 1
                };
                nodes.push(Node::Affine { terms: vec![(gated, select)], bias: None });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, write)], bias: own_means.map(|m| m[j]) });
                projections.push(nodes.len() - 1);
            }
            nodes.push(Node::Attend { query: projections[0], key: projections[1], value: projections[2], scale, rotary, causal });
            let rule = Rule { name: format!("{name}.h{h}"), inputs: inputs.iter().map(|&n| node_interface(n)).collect::<Result<_, _>>()?, output: nodes.len() - 1, nodes };
            artifact = artifact.replace_block(&format!("{name}.h{h}"), Callee::New(rule), inputs.iter().map(|&n| Argument::Native(n)).collect(), read, operators)?;
        }
        // ---------------------------------------------------------------- the attention's output
        let o = site(Kind::Output);
        let o_own = at.get(&(l, 1)).cloned().unwrap_or_default();
        let o_direction = direction_of(&o_own)?;
        // The o slices: those of components gated at the attention's input, then of components
        // gated on their own o read; per o-carrying component its slices.
        let o_cross = carried_into((l, 1));
        let o_carried: usize = o_cross.iter().map(|(_, comps, _)| comps.len()).sum();
        let o_carriers: Vec<usize> = o_cross.iter().flat_map(|(_, comps, _)| comps.iter().copied()).chain(a_comps.iter().copied().filter(|&b| !slices_on(b, o).is_empty())).chain(o_own.iter().copied()).collect();
        if !o_carriers.is_empty() {
            let o_rows: Vec<usize> = o_carriers.iter().flat_map(|&b| slices_on(b, o)).collect();
            let o_widths: Vec<usize> = o_carriers.iter().map(|&b| slices_on(b, o).len()).collect();
            let reads_cols: Vec<Interface> = layer.reads.iter().map(|&r| node_interface(r)).collect::<Result<_, _>>()?;
            let concat = concat_interface(&reads_cols)?;
            let width = concat.width();
            let o_stacked = grouped(&o_widths)?;
            let out_rows = node_interface(layer.attention)?;
            let base = artifact.program.operators.len();
            for (row, &i) in o_rows.iter().enumerate() {
                held.entry((o, i)).or_default().extend([(format!("{name}.o.read"), Held::Read { row }), (format!("{name}.o.write"), Held::Write { column: row, offset: 0 })]);
            }
            let mut operators = vec![
                dense(&format!("{name}.o.read"), o_stacked.clone(), concat.clone(), Array2::from_shape_fn((o_rows.len(), width), |(r, j)| factors[o].v[[j, o_rows[r]]]))?,
                dense(&format!("{name}.o.write"), out_rows.clone(), o_stacked.clone(), Array2::from_shape_fn((out_rows.width(), o_rows.len()), |(r, c)| factors[o].u[[o_rows[c], r]]))?,
            ];
            let heads = layer.reads.len();
            let mut inputs: Vec<usize> = layer.reads.iter().copied().chain([x]).collect();
            inputs.extend(extra_inputs(&o_cross, &inputs));
            let mut nodes: Vec<Node> = (0..inputs.len()).map(|index| Node::Param { index }).collect();
            nodes.push(Node::Concat { parts: (0..heads).collect() });
            let c = nodes.len() - 1;
            // With mean parts (`LayerMeans`): the o slices read the heads' outputs less their mean
            // values, an always-on part writes `W_o E[v]`, and the attention stage's recomputed gate
            // reads `x − E[x]`.
            let o_means = match &attn_means {
                Some((_, heads_means)) => {
                    let Node::Affine { terms, .. } = &native.nodes[layer.attention] else { return Err(error(format!("layer {l}: the attention's output is not one affine map"))) };
                    let mut written = Array1::<f64>::zeros(out_rows.width());
                    for (&read, means) in layer.reads.iter().zip(heads_means) {
                        let (_, op) = terms.iter().find(|(a, _)| *a == read).ok_or_else(|| error(format!("layer {l}: a head the o map does not read")))?;
                        written += &native.operators[*op].matrix().dot(&means[2]);
                    }
                    let values: Vec<f64> = heads_means.iter().flat_map(|m| m[2].iter().copied()).collect();
                    let first = base + operators.len();
                    operators.push(Operator::identity(format!("{name}.o.center"), concat.clone()));
                    operators.push(dense(&format!("{name}.o.center_bias"), concat.clone(), Interface::constant(), Array1::from(values).mapv(|v| -v).insert_axis(Axis(1)))?);
                    operators.push(dense(&format!("{name}.o.mean"), out_rows.clone(), Interface::constant(), written.insert_axis(Axis(1)))?);
                    nodes.push(Node::Affine { terms: vec![(c, first)], bias: Some(first + 1) });
                    let centered = nodes.len() - 1;
                    nodes.push(Node::Affine { terms: vec![(heads, index_of(&artifact.program, &format!("{name}.attn.center"))?)], bias: Some(index_of(&artifact.program, &format!("{name}.attn.center_bias"))?) });
                    Some((centered, nodes.len() - 1, first + 2))
                }
                None => None,
            };
            let o_reads = o_means.map_or(c, |(centered, _, _)| centered);
            // The o reads after their gate when the gate does not read them (no own norm gates), so
            // each row reads only its components on (`DeviceProgram`'s gated reads).
            let late = (!o_own.is_empty() && !o_direction).then(|| {
                nodes.push(Node::Affine { terms: vec![(o_reads, base)], bias: None });
                nodes.len() - 1
            });
            // The gate columns of the o carriers: from the attention input stage's gate (recomputed
            // here from x), then the own o gates.
            let (mut gate_parts, mut soft_parts) = cross_parts(&artifact, &mut nodes, (&mut operators, base), &format!("{name}.o"), &o_cross, &inputs)?;
            let carried: Vec<usize> = o_carriers[o_carried..].iter().filter(|b| a_comps.contains(b)).map(|b| a_comps.iter().position(|a| a == b).unwrap_or(0)).collect();
            if !carried.is_empty() {
                let gate_names = match (a_direction || a_scored, attn_shared) {
                    (true, _) => ("direction", "threshold"),
                    (false, true) => ("assign", "threshold"),
                    (false, false) => ("gate_identity", "threshold"),
                };
                let stage_ops = (
                    index_of(&artifact.program, &format!("{name}.attn.read"))?,
                    index_of(&artifact.program, &format!("{name}.attn.{}", gate_names.0))?,
                    index_of(&artifact.program, &format!("{name}.attn.{}", gate_names.1))?,
                    index_of(&artifact.program, &format!("{name}.attn.width"))?,
                );
                let assign = if attn_shared { Some(index_of(&artifact.program, &format!("{name}.attn.assign"))?) } else { None };
                let own = if a_own { Some(index_of(&artifact.program, &format!("{name}.attn.select_own"))?) } else { None };
                let (_, z, s) = if a_scored {
                    let named = |p: &str| index_of(&artifact.program, &format!("{name}.{p}"));
                    let writes: Vec<(usize, usize)> = (0..heads).map(|h2| Ok((named(&format!("h{h2}.q"))?, named(&format!("h{h2}.k"))?))).collect::<Result<_, String>>()?;
                    let Node::Attend { scale, rotary, causal, .. } = native.nodes[layer.reads[0]].clone() else {
                        return Err(error(format!("layer {l}: a head read is not an attention node")));
                    };
                    let means = match o_means {
                        Some(_) => Some((0..heads).map(|h2| Ok((named(&format!("h{h2}.q_mean"))?, named(&format!("h{h2}.k_mean"))?))).collect::<Result<Vec<_>, String>>()?),
                        None => None,
                    };
                    let x_reads = o_means.map_or(heads, |(_, x_centered, _)| x_centered);
                    let (_, z) = score_gate_nodes(&mut nodes, (x_reads, stage_ops.0, named("attn.select0")?, named("attn.select1")?, named("attn.score_heads")?), (&writes, means.as_deref()), (scale, rotary, causal, widths.len()), (stage_ops.1, stage_ops.2));
                    nodes.push(Node::Constant { operator: stage_ops.3 });
                    (None, z, nodes.len() - 1)
                } else {
                    stage_nodes(&mut nodes, (heads, o_means.map_or(heads, |(_, x_centered, _)| x_centered)), stage_ops, assign, a_direction, own, None, widths.len(), detector_ops(&artifact, &[], 0, &format!("{name}.attn")))
                };
                operators.push(dense(&format!("{name}.o.select_gate"), units(carried.len())?, units(a_comps.len())?, selection(&carried, a_comps.len()))?);
                nodes.push(Node::Affine { terms: vec![(z, base + operators.len() - 1)], bias: None });
                gate_parts.push(nodes.len() - 1);
                nodes.push(Node::Affine { terms: vec![(s, base + operators.len() - 1)], bias: None });
                soft_parts.push(nodes.len() - 1);
            }
            if !o_own.is_empty() {
                // Own o gates read the o carriers' own rows: the group norms of the read's own groups.
                let own_first = o_carried + carried.len();
                operators.push(dense(&format!("{name}.o.select_own"), units(o_own.len())?, units(o_carriers.len())?, selection(&(own_first..o_carriers.len()).collect::<Vec<_>>(), o_carriers.len()))?);
                let select = base + operators.len() - 1;
                if is_shared(&o_own) {
                    return Err(error(format!("layer {l}: gate sharing at the o stage is not built (only at the attention's and the MLP's inputs)")));
                }
                let mut gates = gate_ops(&format!("{name}.o"), &o_own, &concat, false, None, None)?;
                let first = base + operators.len();
                operators.append(&mut gates);
                if o_direction {
                    nodes.push(Node::Affine { terms: vec![(c, first)], bias: Some(first + 1) });
                } else {
                    nodes.push(Node::GroupNorm { input: late.ok_or_else(|| error(format!("layer {l}: own o gates without the o reads")))? });
                    nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![Law::Log; o_carriers.len()] });
                    nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, select)], bias: None });
                    nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, first)], bias: Some(first + 1) });
                }
                gate_parts.push(nodes.len() - 1);
                nodes.push(Node::Constant { operator: first + 2 });
                soft_parts.push(nodes.len() - 1);
            }
            let (z, s) = joined(&mut nodes, gate_parts, soft_parts);
            let a_o = late.unwrap_or_else(|| {
                nodes.push(Node::Affine { terms: vec![(o_reads, base)], bias: None });
                nodes.len() - 1
            });
            nodes.push(Node::Gated { value: a_o, gate: z, scale: Some(s) });
            nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, base + 1)], bias: o_means.map(|(_, _, mean)| mean) });
            let arguments = inputs.iter().map(|&n| Argument::Native(n)).collect();
            let inputs = inputs.iter().map(|&n| node_interface(n)).collect::<Result<_, _>>()?;
            let rule = Rule { name: format!("{name}.o"), inputs, output: nodes.len() - 1, nodes };
            artifact = artifact.replace_block(&format!("{name}.o"), Callee::New(rule), arguments, layer.attention, operators)?;
        }
        // ---------------------------------------------------------------- the MLP
        let (fc, dn) = (site(Kind::Up), site(Kind::Down));
        let f_comps = at.get(&(l, 2)).cloned().unwrap_or_default();
        let d_own = at.get(&(l, 3)).cloned().unwrap_or_default();
        let (f_direction, d_direction, f_active) = (direction_of(&f_comps)?, direction_of(&d_own)?, active_of(&f_comps)?);
        let Node::Pointwise { input: pre, laws } = native.nodes[layer.active].clone() else {
            return Err(error(format!("layer {l}: the MLP activation is not one pointwise law")));
        };
        let up_rows = node_interface(pre)?;
        let h2 = layer.normed;
        let h2_interface = node_interface(h2)?;
        let (f_cross, d_cross) = (carried_into((l, 2)), carried_into((l, 3)));
        let (f_carried, d_carried): (usize, usize) = (f_cross.iter().map(|(_, c, _)| c.len()).sum(), d_cross.iter().map(|(_, c, _)| c.len()).sum());
        if f_carried > 0 && (f_comps.is_empty() || f_active || is_shared(&f_comps)) {
            return Err(error(format!("layer {l}: components carried into an MLP input stage with no own, all-on or shared gates")));
        }
        if f_comps.is_empty() && d_own.is_empty() && f_carried + d_carried == 0 {
            // No component in the MLP (an attention-only model's zero MLP): the block adds zero.
            let out_rows = node_interface(layer.mlp)?;
            let zero = dense(&format!("{name}.mlp.zero"), out_rows.clone(), Interface::constant(), Array2::zeros((out_rows.width(), 1)))?;
            let base = artifact.program.operators.len();
            let rule = Rule { name: format!("{name}.mlp"), inputs: vec![h2_interface.clone()], output: 1, nodes: vec![Node::Param { index: 0 }, Node::Constant { operator: base }] };
            artifact = artifact.replace_block(&format!("{name}.mlp"), Callee::New(rule), vec![Argument::Native(h2)], layer.mlp, vec![zero])?;
            continue;
        }
        let fc_stack: Vec<usize> = f_cross.iter().flat_map(|(_, comps, _)| comps.iter().copied()).chain(f_comps.iter().copied()).collect();
        let fc_rows: Vec<usize> = fc_stack.iter().flat_map(|&b| slices_on(b, fc)).collect();
        let fc_widths: Vec<usize> = fc_stack.iter().map(|&b| slices_on(b, fc).len()).collect();
        if fc_widths.iter().any(|w| *w == 0) || f_comps.is_empty() {
            return Err(error(format!("layer {l}: an MLP component gated at the MLP's input without a c_fc slice")));
        }
        let dn_carriers: Vec<usize> = d_cross.iter().flat_map(|(_, comps, _)| comps.iter().copied()).chain(f_comps.iter().copied().filter(|&b| !slices_on(b, dn).is_empty())).chain(d_own.iter().copied()).collect();
        let dn_rows: Vec<usize> = dn_carriers.iter().flat_map(|&b| slices_on(b, dn)).collect();
        let dn_widths: Vec<usize> = dn_carriers.iter().map(|&b| slices_on(b, dn).len()).collect();
        let out_rows = node_interface(layer.mlp)?;
        let (fc_stacked, dn_stacked) = (grouped(&fc_widths)?, grouped(&dn_widths)?);
        let d2 = h2_interface.width();
        let hidden = up_rows.width();
        let base = artifact.program.operators.len();
        for (row, &i) in fc_rows.iter().enumerate() {
            held.entry((fc, i)).or_default().extend([(format!("{name}.mlp.fc_read"), Held::Read { row }), (format!("{name}.mlp.fc_write"), Held::Write { column: row, offset: 0 })]);
        }
        for (row, &i) in dn_rows.iter().enumerate() {
            held.entry((dn, i)).or_default().extend([(format!("{name}.mlp.dn_read"), Held::Read { row }), (format!("{name}.mlp.dn_write"), Held::Write { column: row, offset: 0 })]);
        }
        let mut operators = vec![
            dense(&format!("{name}.mlp.fc_read"), fc_stacked.clone(), h2_interface.clone(), Array2::from_shape_fn((fc_rows.len(), d2), |(r, j)| factors[fc].v[[j, fc_rows[r]]]))?,
            dense(&format!("{name}.mlp.fc_write"), up_rows.clone(), fc_stacked.clone(), Array2::from_shape_fn((hidden, fc_rows.len()), |(r, c)| factors[fc].u[[fc_rows[c], r]]))?,
            dense(&format!("{name}.mlp.dn_read"), dn_stacked.clone(), up_rows.clone(), Array2::from_shape_fn((dn_rows.len(), hidden), |(r, j)| factors[dn].v[[j, dn_rows[r]]]))?,
            dense(&format!("{name}.mlp.dn_write"), out_rows.clone(), dn_stacked.clone(), Array2::from_shape_fn((out_rows.width(), dn_rows.len()), |(r, c)| factors[dn].u[[dn_rows[c], r]]))?,
        ];
        let mut gates = gate_ops(&format!("{name}.mlp.fc"), &f_comps, &h2_interface, is_shared(&f_comps), detector_set(l, 2), None)?;
        operators.append(&mut gates);
        let mut inputs = vec![h2];
        inputs.extend(extra_inputs(&f_cross, &inputs));
        let d_inputs = extra_inputs(&d_cross, &inputs);
        inputs.extend(d_inputs);
        let mut nodes: Vec<Node> = (0..inputs.len()).map(|index| Node::Param { index }).collect();
        // With mean parts (`LayerMeans`): the c_fc slices read `x − E[x]`, the down slices
        // `act − E[act]`, and `E[p]` and `W_dn E[act]` are written on every token.
        let means = match layer_means.iter().find(|m| m.layer == l && !(m.mlp.is_empty() && m.activation.is_empty())) {
            Some(m) => {
                if m.mlp.len() != d2 || m.activation.len() != hidden || f_active {
                    return Err(error(format!("layer {l}: means of {} and {} for an MLP of {d2} and {hidden}, or beside all-on activation reads", m.mlp.len(), m.activation.len())));
                }
                let Node::Affine { terms, bias } = &native.nodes[pre] else { return Err(error(format!("layer {l}: the MLP's input map is not one affine map"))) };
                let (w_fc, b_fc) = (native.operators[terms[0].1].matrix(), bias.map(|b| native.operators[b].matrix().column(0).to_owned()));
                let w_dn = native.operators[mlp_down(native, layer)?].matrix();
                let (ex, eact) = (Array1::from(m.mlp.clone()), Array1::from(m.activation.clone()));
                let ep = w_fc.dot(&ex) + b_fc.unwrap_or_else(|| Array1::zeros(hidden));
                let column = |v: Array1<f64>| v.insert_axis(Axis(1));
                let first = base + operators.len();
                operators.push(Operator::identity(format!("{name}.mlp.fc_center"), h2_interface.clone()));
                operators.push(dense(&format!("{name}.mlp.fc_center_bias"), h2_interface.clone(), Interface::constant(), column(-&ex))?);
                operators.push(dense(&format!("{name}.mlp.fc_mean"), up_rows.clone(), Interface::constant(), column(ep))?);
                operators.push(Operator::identity(format!("{name}.mlp.dn_center"), up_rows.clone()));
                operators.push(dense(&format!("{name}.mlp.dn_center_bias"), up_rows.clone(), Interface::constant(), column(-&eact))?);
                operators.push(dense(&format!("{name}.mlp.dn_mean"), out_rows.clone(), Interface::constant(), column(w_dn.dot(&eact)))?);
                nodes.push(Node::Affine { terms: vec![(0, first)], bias: Some(first + 1) });
                Some((nodes.len() - 1, first))
            }
            None => None,
        };
        let fc_reads = means.map_or(0, |(centered, _)| centered);
        let fc_parts = cross_parts(&artifact, &mut nodes, (&mut operators, base), &format!("{name}.mlp.fc"), &f_cross, &inputs)?;
        let fc_own = (f_carried > 0 && !f_direction).then(|| {
            let groups = f_carried + f_comps.len();
            operators.push(dense(&format!("{name}.mlp.fc.select_own"), units(f_comps.len())?, units(groups)?, selection(&(f_carried..groups).collect::<Vec<_>>(), groups))?);
            Ok::<usize, String>(base + operators.len() - 1)
        }).transpose()?;
        let fc_assign = is_shared(&f_comps).then(|| if f_direction { base + 7 } else { base + 4 });
        if fc_assign.is_some() {
            shares.push(share_of(&format!("{name}.mlp.fc"), &f_comps)?);
        }
        let (gated, z_f, s_f) = if f_active {
            // The all-on activations ā: every c_fc read, written with every slice on, through the
            // activation law (the c_fc map's second use); per component the norm of its down reads
            // on ā (`{name}.mlp.fc.active`, a copy of those reads held by the gate), less its
            // threshold, or in a shared stage the gates on the squared norms through the assignment
            // (as shared own gates, `stage_nodes`); then the c_fc reads gated.
            let widths: Vec<usize> = f_comps.iter().map(|&b| slices_on(b, dn).len()).collect();
            if widths.contains(&0) {
                return Err(error(format!("layer {l}: an all-on activation read of a component with no down slice")));
            }
            let read: Vec<usize> = f_comps.iter().flat_map(|&b| slices_on(b, dn)).collect();
            operators.push(dense(&format!("{name}.mlp.fc.active"), grouped(&widths)?, up_rows.clone(), Array2::from_shape_fn((read.len(), hidden), |(r, j)| factors[dn].v[[j, read[r]]]))?);
            let active = base + operators.len() - 1;
            nodes.push(Node::Affine { terms: vec![(0, base)], bias: None });
            let a = nodes.len() - 1;
            nodes.push(Node::Affine { terms: vec![(a, base + 1)], bias: None });
            nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: laws.clone() });
            nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, active)], bias: None });
            nodes.push(Node::GroupNorm { input: nodes.len() - 1 });
            if let Some(assign) = fc_assign {
                let norm = nodes.len() - 1;
                nodes.push(Node::Hadamard { left: norm, right: norm });
                nodes.push(Node::Affine { terms: vec![(norm + 1, assign)], bias: Some(base + 5) });
                nodes.push(Node::Constant { operator: base + 6 });
                nodes.push(Node::Transposed { input: norm + 2, operator: assign });
                nodes.push(Node::Transposed { input: norm + 3, operator: assign });
                let (z, s) = (nodes.len() - 2, nodes.len() - 1);
                nodes.push(Node::Gated { value: a, gate: z, scale: Some(s) });
                (nodes.len() - 1, z, s)
            } else {
                nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![Law::Log; f_comps.len()] });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, base + 4)], bias: Some(base + 5) });
                let z = nodes.len() - 1;
                nodes.push(Node::Constant { operator: base + 6 });
                let s = nodes.len() - 1;
                nodes.push(Node::Gated { value: a, gate: z, scale: Some(s) });
                (nodes.len() - 1, z, s)
            }
        } else {
            let (gated, z, s) = stage_nodes(&mut nodes, (0, fc_reads), (base, base + 4, base + 5, base + 6), fc_assign, f_direction, fc_own, Some(fc_parts), fc_widths.len(), detector_ops(&artifact, &operators, base, &format!("{name}.mlp.fc")));
            (gated.ok_or("the gated reads")?, z, s)
        };
        nodes.push(Node::Affine { terms: vec![(gated, base + 1)], bias: means.map(|(_, first)| first + 2) });
        nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: laws.clone() });
        let act = nodes.len() - 1;
        // The down slices' reads: of `act − E[act]` with mean parts.
        let dn_reads = match means {
            Some((_, first)) => {
                nodes.push(Node::Affine { terms: vec![(act, first + 3)], bias: Some(first + 4) });
                nodes.len() - 1
            }
            None => act,
        };
        // The down reads after their gate when the gate does not read them (as the o reads').
        let late = (!d_own.is_empty() && !d_direction).then(|| {
            nodes.push(Node::Affine { terms: vec![(dn_reads, base + 2)], bias: None });
            nodes.len() - 1
        });
        let (mut gate_parts, mut soft_parts) = cross_parts(&artifact, &mut nodes, (&mut operators, base), &format!("{name}.mlp.dn"), &d_cross, &inputs)?;
        let carried: Vec<usize> = dn_carriers[d_carried..].iter().filter(|b| f_comps.contains(b)).map(|b| f_comps.iter().position(|a| a == b).unwrap_or(0)).collect();
        if !carried.is_empty() {
            operators.push(dense(&format!("{name}.mlp.dn_select_gate"), units(carried.len())?, units(f_comps.len())?, selection(&carried, f_comps.len()))?);
            nodes.push(Node::Affine { terms: vec![(z_f, base + operators.len() - 1)], bias: None });
            gate_parts.push(nodes.len() - 1);
            nodes.push(Node::Affine { terms: vec![(s_f, base + operators.len() - 1)], bias: None });
            soft_parts.push(nodes.len() - 1);
        }
        if !d_own.is_empty() {
            let own_first = d_carried + carried.len();
            operators.push(dense(&format!("{name}.mlp.dn_select_own"), units(d_own.len())?, units(dn_carriers.len())?, selection(&(own_first..dn_carriers.len()).collect::<Vec<_>>(), dn_carriers.len()))?);
            let select = base + operators.len() - 1;
            // Down components with candidates follow a c_fc component's gate or keep their own
            // (cross-stage sharing: parts spanning the read and the write maps, as descent's share
            // arm, d6ad2537cb); their own gates stay unshared among themselves.
            let follows = is_shared(&d_own);
            let mut gates = gate_ops(&format!("{name}.mlp.dn"), &d_own, &up_rows, false, None, None)?;
            let first = base + operators.len();
            operators.append(&mut gates);
            if d_direction {
                nodes.push(Node::Affine { terms: vec![(act, first)], bias: Some(first + 1) });
            } else {
                nodes.push(Node::GroupNorm { input: late.ok_or_else(|| error(format!("layer {l}: own down gates without the down reads")))? });
                nodes.push(Node::Pointwise { input: nodes.len() - 1, laws: vec![Law::Log; dn_carriers.len()] });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, select)], bias: None });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, first)], bias: Some(first + 1) });
            }
            let z_own = nodes.len() - 1;
            nodes.push(Node::Constant { operator: first + 2 });
            let s_own = nodes.len() - 1;
            if follows {
                // The assignment over the down components' own gates, then the c_fc components'
                // (`{name}.mlp.dn.assign`, (D + F) × D, its own gate at the start); each down
                // component takes z_b = Σ_m A_mb z_m over both, and its width the same way.
                let (count, fcount) = (d_own.len(), f_comps.len());
                let mut start = Array2::zeros((count + fcount, count));
                for b in 0..count {
                    start[[b, b]] = 1.0;
                }
                operators.push(dense(&format!("{name}.mlp.dn.assign"), concat_interface(&[units(count)?, units(fcount)?])?, units(count)?, start)?);
                let assign = base + operators.len() - 1;
                nodes.push(Node::Concat { parts: vec![z_own, z_f] });
                nodes.push(Node::Concat { parts: vec![s_own, s_f] });
                let (all_z, all_s) = (nodes.len() - 2, nodes.len() - 1);
                nodes.push(Node::Transposed { input: all_z, operator: assign });
                gate_parts.push(nodes.len() - 1);
                nodes.push(Node::Transposed { input: all_s, operator: assign });
                soft_parts.push(nodes.len() - 1);
                let position: BTreeMap<usize, usize> = f_comps.iter().enumerate().map(|(i, b)| (*b, i)).collect();
                let candidates = d_own
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| {
                        let mut list = vec![i];
                        for c in &components[b].candidates {
                            let at = count + *position.get(c).ok_or_else(|| error(format!("down component {b}: candidate {c} is not a c_fc component of its layer")))?;
                            if !list.contains(&at) {
                                list.push(at);
                            }
                        }
                        Ok(list)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                shares.push(Share { operator: format!("{name}.mlp.dn.assign"), candidates });
            } else {
                gate_parts.push(z_own);
                soft_parts.push(s_own);
            }
        }
        let (z, s) = joined(&mut nodes, gate_parts, soft_parts);
        let a_dn = late.unwrap_or_else(|| {
            nodes.push(Node::Affine { terms: vec![(dn_reads, base + 2)], bias: None });
            nodes.len() - 1
        });
        nodes.push(Node::Gated { value: a_dn, gate: z, scale: Some(s) });
        nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, base + 3)], bias: means.map(|(_, first)| first + 5) });
        let rule = Rule { name: format!("{name}.mlp"), inputs: inputs.iter().map(|&n| node_interface(n)).collect::<Result<_, _>>()?, output: nodes.len() - 1, nodes };
        artifact = artifact.replace_block(&format!("{name}.mlp"), Callee::New(rule), inputs.iter().map(|&n| Argument::Native(n)).collect(), layer.mlp, operators)?;
    }
    // Each shared component's gate is a choice among its candidates: ln K nats to send.
    let choices: f64 = shares.iter().flat_map(|s| s.candidates.iter()).map(|c| (c.len() as f64).ln()).sum();
    let mixes = mixing.iter().map(|group| mix_of(&artifact.program, &held, group, &starts)).collect::<Result<Vec<_>, String>>()?;
    let built = groups_of(artifact, layers, gate)?;
    let scoring = match gate {
        Gate::Hard => crate::library_mdl::GateScoring::Compiled,
        Gate::Learned => crate::library_mdl::GateScoring::Hard,
    };
    Ok(Explanation { shares, fixed_nats: built.fixed_nats + choices, scoring, mixes, ..built })
}

/// The frame of one group of slices `group` (`[site, index]`, every slice of one map, its start
/// slices, those below the site's count in `starts`, before its extra slices): per slice its places
/// ([`Held`], resolved to `program`'s operators), in one order for every slice, so the group's
/// reads stack into one matrix and each of its write blocks into another.
fn mix_of(program: &OperatorProgram, held: &BTreeMap<(usize, usize), Vec<(String, Held)>>, group: &[[usize; 2]], starts: &[usize]) -> Result<crate::library_mdl::Mix, String> {
    let site = group.first().ok_or_else(|| error("an empty mixing group"))?[0];
    let start = *starts.get(site).ok_or_else(|| error(format!("a mixing group at site {site}, of {}", starts.len())))?;
    let base = group.iter().take_while(|s| s[1] < start).count();
    if group[base..].iter().any(|s| s[1] < start) {
        return Err(error(format!("a mixing group of site {site} with a start slice after an extra slice")));
    }
    let mut slices = Vec::with_capacity(group.len());
    let mut shape: Option<Vec<(usize, bool, usize)>> = None;
    for &[s, i] in group {
        if s != site {
            return Err(error(format!("a mixing group across sites {site} and {s}: a group's frame is one map's")));
        }
        let places = held.get(&(s, i)).ok_or_else(|| error(format!("slice {i} of site {s} is in no component")))?;
        let mut resolved = places
            .iter()
            .map(|(name, place)| {
                let operator = index_of(program, name)?;
                Ok(match *place {
                    Held::Read { row } => crate::library_mdl::Place::Read { operator, row },
                    Held::Write { column, offset } => crate::library_mdl::Place::Write { operator, column, offset },
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        resolved.sort_by_key(crate::library_mdl::Place::key);
        let this: Vec<(usize, bool, usize)> = resolved.iter().map(crate::library_mdl::Place::key).collect();
        match &shape {
            None => shape = Some(this),
            Some(first) if *first == this => {}
            Some(_) => return Err(error(format!("slice {i} of site {s} is held unlike the rest of its mixing group"))),
        }
        slices.push(resolved);
    }
    let mut seen = std::collections::BTreeSet::new();
    if !slices.iter().flatten().all(|p| seen.insert(p.clone())) {
        return Err(error(format!("a slice listed twice in a mixing group of site {site}")));
    }
    Ok(crate::library_mdl::Mix { slices, base })
}

/// The prior groups, trainable operators and layers of the built artifact (module note).
fn groups_of(artifact: Artifact, layers: &[LayerNodes], gate: Gate) -> Result<Explanation, String> {
    let program = &artifact.program;
    let named = |name: &str| index_of(program, name);
    let mut groups: Vec<Group> = Vec::new();
    // A detector stage's combination entries off their gates' supports, zero and removed.
    let mut removed: Vec<usize> = Vec::new();
    // A stage's gate rows (`{prefix}.direction`, one group per row, `g{b}`), and with detectors their
    // rows and thresholds; each group's index into `into`.
    let mut gate_groups = |prefix: &str, groups: &mut Vec<Group>, trainable: &mut Vec<usize>, into: &mut Vec<usize>| -> Result<(), String> {
        let Ok(g) = named(&format!("{prefix}.direction")) else { return Ok(()) };
        trainable.push(g);
        let (rows, cols) = (program.operators[g].rows.width(), program.operators[g].cols.width());
        let detectors = named(&format!("{prefix}.detectors")).ok();
        let sparse = detectors.is_some() || named(&format!("{prefix}.score_heads")).is_ok();
        let values = program.operators[g].matrix();
        let mut off = Vec::new();
        for b in 0..rows {
            let cells = match sparse {
                true => {
                    off.extend((0..cols).filter(|j| values[[b, *j]] == 0.0).map(|j| Cells { operator: g, rows: vec![b], cols: j..j + 1 }));
                    (0..cols).filter(|j| values[[b, *j]] != 0.0).map(|j| Cells { operator: g, rows: vec![b], cols: j..j + 1 }).collect()
                }
                false => vec![Cells { operator: g, rows: vec![b], cols: 0..cols }],
            };
            groups.push(Group { name: format!("{prefix}.g{b}"), cells });
            into.push(groups.len() - 1);
        }
        if !off.is_empty() {
            groups.push(Group { name: format!("{prefix}.off"), cells: off });
            removed.push(groups.len() - 1);
        }
        if let Some(d) = detectors {
            trainable.push(d);
            for j in 0..program.operators[d].rows.width() {
                groups.push(Group { name: format!("{prefix}.det{j}"), cells: vec![Cells { operator: d, rows: vec![j], cols: 0..program.operators[d].cols.width() }] });
                into.push(groups.len() - 1);
            }
            let t = named(&format!("{prefix}.detector_thresholds"))?;
            trainable.push(t);
            groups.push(Group { name: format!("{prefix}.detector_thresholds"), cells: vec![Cells { operator: t, rows: (0..program.operators[t].rows.width()).collect(), cols: 0..1 }] });
            into.push(groups.len() - 1);
        }
        Ok(())
    };
    let mut trainable: Vec<usize> = Vec::new();
    let mut out: Vec<Layer> = layers.iter().map(|sites| Layer { sites: sites.clone(), heads: Vec::new(), functions: Vec::new(), sink: None, thresholds: Vec::new(), components: Vec::new() }).collect();
    // A group per row (or column) of an operator, named.
    let rows_of = |op: usize| program.operators[op].rows.width();
    let cols_of = |op: usize| program.operators[op].cols.width();
    for (l, layer) in out.iter_mut().enumerate() {
        let name = format!("library.l{l}");
        let heads = layers[l].reads.len();
        // The attention input stage: each stacked read row with its write columns in every head.
        let read = named(&format!("{name}.attn.read"))?;
        trainable.push(read);
        let mut per_component: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let stacked = program.operators[read].rows.clone();
        let mut row = 0;
        let selects: Vec<Option<usize>> = (0..3).map(|j| named(&format!("{name}.attn.select{j}")).ok()).collect();
        // Per stacked row: its map (q, k, v) and its column in that map's writes.
        let mut column_in: Vec<(usize, usize)> = Vec::new();
        let mut counters = [0usize; 3];
        let select_values: Vec<Option<Array2<f64>>> = selects.iter().map(|s| s.map(|op| program.operators[op].matrix())).collect();
        for r in 0..rows_of(read) {
            let j = select_values.iter().position(|m| m.as_ref().is_some_and(|m| m.column(r).iter().any(|v| *v != 0.0))).ok_or_else(|| error("a stacked row in no selection"))?;
            column_in.push((j, counters[j]));
            counters[j] += 1;
        }
        for (b, group) in stacked.groups().iter().enumerate() {
            for _ in 0..group.width {
                let (j, c) = column_in[row];
                let mut cells = vec![];
                groups.push(Group { name: format!("{name}.attn.c{b}.r{row}.read"), cells: vec![Cells { operator: read, rows: vec![row], cols: 0..cols_of(read) }] });
                per_component.entry(b).or_default().push(groups.len() - 1);
                for h in 0..heads {
                    let write = named(&format!("{name}.h{h}.{}", ["q", "k", "v"][j]))?;
                    cells.push(Cells { operator: write, rows: (0..rows_of(write)).collect(), cols: c..c + 1 });
                }
                groups.push(Group { name: format!("{name}.attn.c{b}.r{row}.write"), cells });
                per_component.entry(b).or_default().push(groups.len() - 1);
                row += 1;
            }
        }
        for h in 0..heads {
            for part in ["q", "k", "v"] {
                trainable.push(named(&format!("{name}.h{h}.{part}"))?);
            }
        }
        // The o slices: read rows and write columns; per carrier its component index in the stage.
        if let Ok(o_read) = named(&format!("{name}.o.read")) {
            let o_write = named(&format!("{name}.o.write"))?;
            trainable.extend([o_read, o_write]);
            let o_groups = program.operators[o_read].rows.clone();
            let mut r = 0;
            for (k, group) in o_groups.groups().iter().enumerate() {
                let mut mine = Vec::new();
                for _ in 0..group.width {
                    groups.push(Group { name: format!("{name}.o.k{k}.r{r}.read"), cells: vec![Cells { operator: o_read, rows: vec![r], cols: 0..cols_of(o_read) }] });
                    mine.push(groups.len() - 1);
                    groups.push(Group { name: format!("{name}.o.k{k}.r{r}.write"), cells: vec![Cells { operator: o_write, rows: (0..rows_of(o_write)).collect(), cols: r..r + 1 }] });
                    mine.push(groups.len() - 1);
                    r += 1;
                }
                layer.components.push(mine);
            }
        }
        layer.components.extend(per_component.into_values());
        // The attention's gates: direction rows, and the thresholds of each stage as one group.
        let mut thresholds = Vec::new();
        for prefix in ["attn", "o"] {
            // A direction stage's gate rows (only such a stage has a direction operator).
            gate_groups(&format!("{name}.{prefix}"), &mut groups, &mut trainable, &mut thresholds)?;
            if let Ok(t) = named(&format!("{name}.{prefix}.threshold")) {
                trainable.push(t);
                groups.push(Group { name: format!("{name}.{prefix}.thresholds"), cells: vec![Cells { operator: t, rows: (0..rows_of(t)).collect(), cols: 0..1 }] });
                thresholds.push(groups.len() - 1);
            }
            if let Ok(w) = named(&format!("{name}.{prefix}.width"))
                && gate != Gate::Hard
            {
                trainable.push(w);
                groups.push(Group { name: format!("{name}.{prefix}.widths"), cells: vec![Cells { operator: w, rows: (0..rows_of(w)).collect(), cols: 0..1 }] });
                thresholds.push(groups.len() - 1);
            }
        }
        layer.components.push(thresholds);
        // The MLP: per carrier of c_fc and down slices, its groups (none for an MLP with no
        // component, which adds zero).
        if named(&format!("{name}.mlp.zero")).is_ok() {
            continue;
        }
        let (fc_read, fc_write, dn_read, dn_write) = (named(&format!("{name}.mlp.fc_read"))?, named(&format!("{name}.mlp.fc_write"))?, named(&format!("{name}.mlp.dn_read"))?, named(&format!("{name}.mlp.dn_write"))?);
        trainable.extend([fc_read, fc_write, dn_read, dn_write]);
        let mut r = 0;
        for (b, group) in program.operators[fc_read].rows.clone().groups().iter().enumerate() {
            let mut mine = Vec::new();
            for _ in 0..group.width {
                groups.push(Group { name: format!("{name}.mlp.c{b}.fc{r}.read"), cells: vec![Cells { operator: fc_read, rows: vec![r], cols: 0..cols_of(fc_read) }] });
                mine.push(groups.len() - 1);
                groups.push(Group { name: format!("{name}.mlp.c{b}.fc{r}.write"), cells: vec![Cells { operator: fc_write, rows: (0..rows_of(fc_write)).collect(), cols: r..r + 1 }] });
                mine.push(groups.len() - 1);
                r += 1;
            }
            layer.functions.push(mine);
        }
        let mut r = 0;
        for (k, group) in program.operators[dn_read].rows.clone().groups().iter().enumerate() {
            let mut mine = Vec::new();
            for _ in 0..group.width {
                groups.push(Group { name: format!("{name}.mlp.k{k}.dn{r}.read"), cells: vec![Cells { operator: dn_read, rows: vec![r], cols: 0..cols_of(dn_read) }] });
                mine.push(groups.len() - 1);
                groups.push(Group { name: format!("{name}.mlp.k{k}.dn{r}.write"), cells: vec![Cells { operator: dn_write, rows: (0..rows_of(dn_write)).collect(), cols: r..r + 1 }] });
                mine.push(groups.len() - 1);
                r += 1;
            }
            layer.functions.push(mine);
        }
        // An all-on activation stage's gate reads, one group per component.
        if let Ok(g) = named(&format!("{name}.mlp.fc.active")) {
            trainable.push(g);
            let mut r = 0;
            for (b, group) in program.operators[g].rows.clone().groups().iter().enumerate() {
                groups.push(Group { name: format!("{name}.mlp.fc.active{b}"), cells: vec![Cells { operator: g, rows: (r..r + group.width).collect(), cols: 0..cols_of(g) }] });
                layer.thresholds.push(groups.len() - 1);
                r += group.width;
            }
        }
        for prefix in ["fc", "dn"] {
            gate_groups(&format!("{name}.mlp.{prefix}"), &mut groups, &mut trainable, &mut layer.thresholds)?;
            if let Ok(t) = named(&format!("{name}.mlp.{prefix}.threshold")) {
                trainable.push(t);
                groups.push(Group { name: format!("{name}.mlp.{prefix}.thresholds"), cells: vec![Cells { operator: t, rows: (0..rows_of(t)).collect(), cols: 0..1 }] });
                layer.thresholds.push(groups.len() - 1);
            }
            if let Ok(w) = named(&format!("{name}.mlp.{prefix}.width"))
                && gate != Gate::Hard
            {
                trainable.push(w);
                groups.push(Group { name: format!("{name}.mlp.{prefix}.widths"), cells: vec![Cells { operator: w, rows: (0..rows_of(w)).collect(), cols: 0..1 }] });
                layer.thresholds.push(groups.len() - 1);
            }
        }
    }
    trainable.sort_unstable();
    trainable.dedup();
    let reference = mean_squares(&artifact.program, &groups);
    Ok(Explanation { artifact, trainable, groups, layers: out, removed, fixed_nats: 0.0, reference, reads: Vec::new(), shares: Vec::new(), scoring: crate::library_mdl::GateScoring::Compiled, mixes: Vec::new() })
}

/// The uses of `M`'s maps in an explanation [`explanation`] built (`artifact::Owner::uses`), where
/// each map is a sum of gated slices no single operator holds: per layer, each head's q, k and v map
/// (the head rule's input into the node writing the head's projection), each head's block of the o
/// map (the heads' concatenated outputs into the o rule's output), the c_fc map (the MLP rule's input
/// into its pre-activations) and the down map (the activations into the MLP's output). A native
/// weight edit compiles through them (`weight_edit`). Empty for an artifact this module did not
/// build. `explanation` does not record them in the artifact, so the identities of its checkpoints
/// stay as they are; whoever compiles an edit adds them.
pub fn uses(native: &OperatorProgram, layers: &[LayerNodes], artifact: &Artifact) -> Result<Vec<Owner>, String> {
    let program = &artifact.program;
    if index_of(program, "library.l0.attn.read").is_err() {
        return Ok(Vec::new());
    }
    let rule = |name: &str| program.rules.iter().find(|r| r.name == name).ok_or_else(|| error(format!("no rule {name}")));
    let applying = |body: &Rule, op: &str| -> Result<Vec<usize>, String> {
        let at = index_of(program, op)?;
        Ok(body.nodes.iter().enumerate().filter(|(_, n)| matches!(n, Node::Affine { terms, .. } if terms.iter().any(|t| t.1 == at))).map(|(i, _)| i).collect())
    };
    let map_of = |node: usize, input: usize| -> Result<&Operator, String> {
        match native.nodes.get(node) {
            Some(Node::Affine { terms, .. }) => terms.iter().find(|t| t.0 == input).map(|t| native.operators[t.1].as_ref()).ok_or_else(|| error(format!("native node {node} reads no map of node {input}"))),
            _ => Err(error(format!("native node {node} is not a map"))),
        }
    };
    let owner = |body: &Rule, w: &Operator, cols: std::ops::Range<usize>, role: &str, at: (usize, usize)| Owner {
        rows: 0..w.rows.width(),
        cols,
        body: body.name.clone(),
        site: body.name.clone(),
        native: w.name.clone(),
        native_rows: 0..w.rows.width(),
        native_cols: 0..w.cols.width(),
        role: role.to_string(),
        uses: Some(at),
        ..Owner::default()
    };
    let mut out = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        let name = format!("library.l{l}");
        // Each head rule's projections, and with logit reads every head's q and k on (`Read::Logits`).
        for h in 0..layer.reads.len() {
            let body = rule(&format!("{name}.h{h}"))?;
            for (h2, &read) in layer.reads.iter().enumerate() {
                let Node::Attend { query, key, value, .. } = &native.nodes[read] else { return Err(error(format!("layer {l} head {h2}: the read is not an attention node"))) };
                for (part, node) in ["q", "k", "v"].into_iter().zip([*query, *key, *value]) {
                    let w = map_of(node, layer.normed_stream)?;
                    let nodes = applying(body, &format!("{name}.h{h2}.{part}"))?;
                    if h2 == h && nodes.is_empty() {
                        return Err(error(format!("{}: no node applies its {part} map", body.name)));
                    }
                    for n in nodes {
                        out.push(owner(body, w, 0..w.cols.width(), part, (0, n)));
                    }
                }
            }
        }
        if let Ok(body) = rule(&format!("{name}.o")) {
            let concat = body.nodes.iter().position(|n| matches!(n, Node::Concat { .. })).ok_or_else(|| error(format!("{}: no concatenation of the heads", body.name)))?;
            let mut offset = 0;
            for &read in &layer.reads {
                let w = map_of(layer.attention, read)?;
                let width = w.cols.width();
                out.push(owner(body, w, offset..offset + width, "o", (concat, body.output)));
                offset += width;
            }
        }
        let body = rule(&format!("{name}.mlp"))?;
        // An MLP with no component adds zero: no slice of c_fc or down is P's.
        if !body.nodes.iter().any(|n| matches!(n, Node::Pointwise { .. })) {
            continue;
        }
        let Node::Pointwise { input: pre, .. } = native.nodes[layer.active] else { return Err(error(format!("layer {l}: the MLP activation is not one pointwise law"))) };
        let fc = map_of(pre, layer.normed)?;
        // Every use of the c_fc map (an all-on activation read's too, `Read::Active`).
        let fc_write = index_of(program, &format!("{name}.mlp.fc_write"))?;
        for (n, _) in body.nodes.iter().enumerate().filter(|(_, n)| matches!(n, Node::Affine { terms, .. } if terms.iter().any(|t| t.1 == fc_write))) {
            out.push(owner(body, fc, 0..fc.cols.width(), "gate", (0, n)));
        }
        // The down map's input: the activations its reads take.
        let dn_read = index_of(program, &format!("{name}.mlp.dn_read"))?;
        let act = body.nodes.iter().find_map(|n| if let Node::Affine { terms, .. } = n { terms.iter().find(|t| t.1 == dn_read).map(|t| t.0) } else { None }).ok_or_else(|| error(format!("{}: no activations", body.name)))?;
        let down = map_of(layer.mlp, layer.active)?;
        out.push(owner(body, down, 0..down.cols.width(), "out", (act, body.output)));
    }
    Ok(out)
}

/// The gate and width nodes of a write-side stage from their parts (carried, then own): the part
/// itself when one, else their concatenation.
fn joined(nodes: &mut Vec<Node>, gates: Vec<usize>, softs: Vec<usize>) -> (usize, usize) {
    if gates.len() == 1 {
        return (gates[0], softs[0]);
    }
    nodes.push(Node::Concat { parts: gates });
    nodes.push(Node::Concat { parts: softs });
    (nodes.len() - 2, nodes.len() - 1)
}

/// The interface of the concatenation of `parts`.
fn concat_interface(parts: &[Interface]) -> Result<Interface, String> {
    Interface::new(parts.iter().flat_map(|p| p.groups().to_vec()).collect()).map_err(error)
}

/// `M`'s down map of `layer`'s MLP: the affine map from its activations to its output.
fn mlp_down(native: &OperatorProgram, layer: &LayerNodes) -> Result<usize, String> {
    match &native.nodes[layer.mlp] {
        Node::Affine { terms, .. } if terms.len() == 1 && terms[0].0 == layer.active => Ok(terms[0].1),
        other => Err(error(format!("the MLP's output is {other:?}, not one map of its activations"))),
    }
}

/// A score stage's heads' largest scores, one column each (`Node::MaxScore`, concatenated).
fn score_heads(heads: usize) -> Result<Interface, String> {
    concat_interface(&vec![units(1)?; heads])
}

/// A score stage's gate features: its `count` groups' log norms, then its heads' largest scores.
fn score_features(count: usize, heads: usize) -> Result<Interface, String> {
    concat_interface(&[units(count)?, score_heads(heads)?])
}

/// A score stage's gate pre-activations (`ScoreRead`) from its input `x` (less its mean with mean
/// parts, whose heads' q and k means `means` the scores' queries and keys add): its stacked reads
/// (`read`), their `groups` group norms' logs and, per head (its q and k writes in `writes`, through
/// the selections `select_q` and `select_k`), the largest score of its query over its keys with
/// every q and k slice on (`Node::MaxScore`; `score_heads` the identity marking the stage), through
/// the gate rows `gate` and the thresholds `threshold`; returns (the stacked reads, the gate node).
fn score_gate_nodes(nodes: &mut Vec<Node>, (x, read, select_q, select_k, score_heads): (usize, usize, usize, usize, usize), (writes, means): (&[(usize, usize)], Option<&[(usize, usize)]>), (scale, rotary, causal, groups): (Scale, Option<Rotary>, bool, usize), (gate, threshold): (usize, usize)) -> (usize, usize) {
    let push = |nodes: &mut Vec<Node>, node: Node| {
        nodes.push(node);
        nodes.len() - 1
    };
    let a = push(nodes, Node::Affine { terms: vec![(x, read)], bias: None });
    let norms = push(nodes, Node::GroupNorm { input: a });
    let logs = push(nodes, Node::Pointwise { input: norms, laws: vec![Law::Log; groups] });
    let sq = push(nodes, Node::Affine { terms: vec![(a, select_q)], bias: None });
    let sk = push(nodes, Node::Affine { terms: vec![(a, select_k)], bias: None });
    let mut scores = Vec::with_capacity(writes.len());
    for (h, &(wq, wk)) in writes.iter().enumerate() {
        let mean = means.map(|m| m[h]);
        let qh = push(nodes, Node::Affine { terms: vec![(sq, wq)], bias: mean.map(|m| m.0) });
        let kh = push(nodes, Node::Affine { terms: vec![(sk, wk)], bias: mean.map(|m| m.1) });
        scores.push(push(nodes, Node::MaxScore { query: qh, key: kh, scale, rotary, causal }));
    }
    let joined = push(nodes, Node::Concat { parts: scores });
    let marked = push(nodes, Node::Affine { terms: vec![(joined, score_heads)], bias: None });
    let features = push(nodes, Node::Concat { parts: vec![logs, marked] });
    (a, push(nodes, Node::Affine { terms: vec![(features, gate)], bias: Some(threshold) }))
}

/// A logit-read stage's constants ([`Read::Logits`], [`logit_nodes`]), named `{prefix}.spread.*`,
/// for `heads` heads of key rows `key` (a head's width `d_h`), with the attention's rotation and
/// scale `c`, over the stage's stacked read rows `stacked` (components of `widths`, `logit` the
/// rows of logit reads, `u(r, j)` row `r`'s write at q coordinate `j` of every head):
/// - `swap` each plane's two coordinates exchanged; `pick{j}` the `j`-th of the pattern's
///   moments (the keys turned to their positions, their squares, with a rotation each plane's
///   product); `neg` minus the identity;
/// - with a rotation, a plane `(a, b)`'s variances `P` and covariance `X` to `w = ((P_a − P_b)/2,
///   X_a)` (`half`, `cross`), and `D = ((P_a + P_b)/2 + r_a, (P_a + P_b)/2 − r_a)` from `P` and
///   `r = R_t⁻² w` (`mean`, `pm`): the keys' covariance turned to the query's frame, `R_tᵀ C R_t`,
///   on its diagonal (a double angle, so two turns back);
/// - per head `h{h}` the spread's weights `u_ik² c²` (stacked rows × key rows, zero off logit
///   rows); `unit` one on the other rows; `sum` each component's rows summed.
fn spread_ops(prefix: &str, (key, heads): (&Interface, usize), (rotary, c): (Option<Rotary>, f64), (stacked, widths, logit): (&Interface, &[usize], &[bool]), u: &dyn Fn(usize, usize) -> f64) -> Result<Vec<Operator>, String> {
    let (hd, r_a) = (key.width(), stacked.width());
    let pairs: Vec<(usize, usize)> = rotary.map(|r| r.pairs()).unwrap_or_default();
    let square = |f: &dyn Fn(&mut Array2<f64>)| {
        let mut m = Array2::zeros((hd, hd));
        f(&mut m);
        m
    };
    let moments = if rotary.is_some() { 3 } else { 2 };
    let joined = concat_interface(&vec![key.clone(); moments])?;
    let mut ops = Vec::new();
    for j in 0..moments {
        ops.push(dense(&format!("{prefix}.spread.pick{j}"), key.clone(), joined.clone(), Array2::from_shape_fn((hd, moments * hd), |(r, col)| f64::from(u8::from(col == j * hd + r))))?);
    }
    ops.push(dense(&format!("{prefix}.spread.neg"), key.clone(), key.clone(), -Array2::<f64>::eye(hd))?);
    if rotary.is_some() {
        let entries: [(&str, &dyn Fn(&mut Array2<f64>, usize, usize)); 5] = [
            ("swap", &|m, a, b| {
                m[[a, b]] = 1.0;
                m[[b, a]] = 1.0;
            }),
            ("half", &|m, a, b| {
                m[[a, a]] = 0.5;
                m[[a, b]] = -0.5;
            }),
            ("cross", &|m, a, b| m[[b, a]] = 1.0),
            ("mean", &|m, a, b| {
                for (i, j) in [(a, a), (a, b), (b, a), (b, b)] {
                    m[[i, j]] = 0.5;
                }
            }),
            ("pm", &|m, a, b| {
                m[[a, a]] = 1.0;
                m[[b, a]] = -1.0;
            }),
        ];
        for (name, set) in entries {
            ops.push(dense(&format!("{prefix}.spread.{name}"), key.clone(), key.clone(), square(&|m| pairs.iter().for_each(|&(a, b)| set(m, a, b))))?);
        }
    }
    for h in 0..heads {
        ops.push(dense(&format!("{prefix}.spread.h{h}"), stacked.clone(), key.clone(), Array2::from_shape_fn((r_a, hd), |(r, j)| if logit[r] { u(r, h * hd + j).powi(2) * c * c } else { 0.0 }))?);
    }
    ops.push(dense(&format!("{prefix}.spread.unit"), stacked.clone(), Interface::constant(), Array2::from_shape_fn((r_a, 1), |(r, _)| f64::from(u8::from(!logit[r]))))?);
    let mut sum = Array2::zeros((widths.len(), r_a));
    let mut row = 0;
    for (b, &w) in widths.iter().enumerate() {
        sum.slice_mut(s![b, row..row + w]).fill(1.0);
        row += w;
    }
    ops.push(dense(&format!("{prefix}.spread.sum"), units(widths.len())?, stacked.clone(), sum)?);
    Ok(ops)
}

/// A logit-read stage's gate inputs ([`Read::Logits`]) from its input `x` (`read` its stacked read,
/// `select_q` and `select_k` its q and k rows): per head (its q and k writes in `writes`) the
/// pattern with every q and k slice on, its moments of the keys turned to their positions, their
/// variances turned to the query's frame (`spread_ops`; floored at zero, as rounding may take
/// them below), then per stacked row its weight (the spread's on a logit read's row, one
/// elsewhere) and per component `Σ_i weight_i c_i²`; returns (the stacked reads, the sums), with
/// `part` the stage's operators by name.
fn logit_nodes(nodes: &mut Vec<Node>, (x, read, select_q, select_k): (usize, usize, usize, usize), writes: &[(usize, usize)], (scale, rotary, causal, groups): (Scale, Option<Rotary>, bool, usize), part: &dyn Fn(&str) -> Result<usize, String>) -> Result<(usize, usize), String> {
    let push = |nodes: &mut Vec<Node>, node: Node| {
        nodes.push(node);
        nodes.len() - 1
    };
    let a = push(nodes, Node::Affine { terms: vec![(x, read)], bias: None });
    let sq = push(nodes, Node::Affine { terms: vec![(a, select_q)], bias: None });
    let sk = push(nodes, Node::Affine { terms: vec![(a, select_k)], bias: None });
    let (neg, pick) = (part("spread.neg")?, |j: usize| part(&format!("spread.pick{j}")));
    let mut terms = Vec::new();
    for (h, &(wq, wk)) in writes.iter().enumerate() {
        let qh = push(nodes, Node::Affine { terms: vec![(sq, wq)], bias: None });
        let kh = push(nodes, Node::Affine { terms: vec![(sk, wk)], bias: None });
        let kr = match rotary {
            Some(r) => push(nodes, Node::Rotate { input: kh, rotary: r, inverse: false }),
            None => kh,
        };
        let mut moments = vec![kr, push(nodes, Node::Hadamard { left: kr, right: kr })];
        if rotary.is_some() {
            let swapped = push(nodes, Node::Affine { terms: vec![(kr, part("spread.swap")?)], bias: None });
            moments.push(push(nodes, Node::Hadamard { left: kr, right: swapped }));
        }
        let value = push(nodes, Node::Concat { parts: moments });
        let st = push(nodes, Node::Attend { query: qh, key: kh, value, scale, rotary, causal });
        let mean = push(nodes, Node::Affine { terms: vec![(st, pick(0)?)], bias: None });
        let squared = push(nodes, Node::Hadamard { left: mean, right: mean });
        let variance = push(nodes, Node::Affine { terms: vec![(st, pick(1)?), (squared, neg)], bias: None });
        let spread = match rotary {
            Some(r) => {
                let swapped = push(nodes, Node::Affine { terms: vec![(mean, part("spread.swap")?)], bias: None });
                let product = push(nodes, Node::Hadamard { left: mean, right: swapped });
                let covariance = push(nodes, Node::Affine { terms: vec![(st, pick(2)?), (product, neg)], bias: None });
                let w = push(nodes, Node::Affine { terms: vec![(variance, part("spread.half")?), (covariance, part("spread.cross")?)], bias: None });
                let back = push(nodes, Node::Rotate { input: w, rotary: r, inverse: true });
                let twice = push(nodes, Node::Rotate { input: back, rotary: r, inverse: true });
                push(nodes, Node::Affine { terms: vec![(variance, part("spread.mean")?), (twice, part("spread.pm")?)], bias: None })
            }
            None => variance,
        };
        let floored = push(nodes, Node::Pointwise { input: spread, laws: vec![crate::operator_program::Law::Relu; groups] });
        terms.push((floored, part(&format!("spread.h{h}"))?));
    }
    let weight = push(nodes, Node::Affine { terms, bias: Some(part("spread.unit")?) });
    let squares = push(nodes, Node::Hadamard { left: a, right: a });
    let weighted = push(nodes, Node::Hadamard { left: squares, right: weight });
    let sum = push(nodes, Node::Affine { terms: vec![(weighted, part("spread.sum")?)], bias: None });
    Ok((a, sum))
}

/// `M`'s operator of each site kind, in [`KINDS`] order, by its export name.
pub const EXPORT_NAMES: [&str; 6] = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"];

/// One component of [`dump_parts`]: per operator of `M` its slices' writes and reads (as columns),
/// its gate, a direction gate's `g`, and its hard gate on every row.
struct Dumped {
    slices: BTreeMap<String, (Vec<Vec<f64>>, Vec<Vec<f64>>)>,
    gate: serde_json::Value,
    g: Option<Vec<f64>>,
    on: Vec<bool>,
}

/// A stage's gate of one component: on per row, `τ_b`, and a direction gate's `g`.
type StageGate = (Vec<bool>, f64, Option<Vec<f64>>);

/// One gate of a stage in [`dump_parts`]'s report: its member components (positions in the
/// stage), the share of rows it is on, and the distinct contexts it fires in ([`contexts`]).
type GateReport = (Vec<usize>, f64, usize);

/// The distinct contexts a part fires in: k-means of its firing rows' reads (`points`, rows × the
/// part's read width; at most 4,096 rows, every ⌈n / 4096⌉-th), `k` from 1 to 8 chosen by the
/// Bayesian information criterion of a spherical Gaussian mixture with one shared variance,
/// `n r ln(SSE / (n r)) + k (r + 1) ln n` (descent's report, d6ad2537cb); 0 for a part never on.
fn contexts(points: &Array2<f64>) -> usize {
    let total = points.nrows();
    if total == 0 {
        return 0;
    }
    let rows: Vec<usize> = (0..total).step_by(total.div_ceil(4096)).collect();
    let p = points.select(Axis(0), &rows);
    let (n, r) = p.dim();
    if n < 2 || r == 0 {
        return 1;
    }
    let mut best = (f64::INFINITY, 1);
    for k in 1..=n.min(8) {
        let sse = kmeans_sse(&p, k);
        let bic = if sse > 0.0 { (n * r) as f64 * (sse / (n * r) as f64).ln() + (k * (r + 1)) as f64 * (n as f64).ln() } else { f64::NEG_INFINITY };
        if bic < best.0 {
            best = (bic, k);
        }
        if sse <= 0.0 {
            break;
        }
    }
    best.1
}

/// Lloyd's k-means of the rows of `p` from a farthest-point start (the row nearest the mean, then
/// each next center the row farthest from the centers so far), at most 50 iterations: the sum of
/// squared distances to the final centers.
fn kmeans_sse(p: &Array2<f64>, k: usize) -> f64 {
    let n = p.nrows();
    let distance = |i: usize, c: &Array1<f64>| -> f64 {
        let d = &p.row(i) - c;
        d.dot(&d)
    };
    let Some(mean) = p.mean_axis(Axis(0)) else { return 0.0 };
    let first = (0..n).min_by(|a, b| distance(*a, &mean).total_cmp(&distance(*b, &mean))).unwrap_or(0);
    let mut centers = vec![p.row(first).to_owned()];
    let mut nearest: Vec<f64> = (0..n).map(|i| distance(i, &centers[0])).collect();
    while centers.len() < k {
        let far = (0..n).max_by(|a, b| nearest[*a].total_cmp(&nearest[*b])).unwrap_or(0);
        centers.push(p.row(far).to_owned());
        for (i, d) in nearest.iter_mut().enumerate() {
            *d = d.min(distance(i, &centers[centers.len() - 1]));
        }
    }
    let closest = |i: usize, centers: &[Array1<f64>]| -> (usize, f64) { centers.iter().enumerate().map(|(j, c)| (j, distance(i, c))).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((0, 0.0)) };
    let mut member = vec![usize::MAX; n];
    for _ in 0..50 {
        let mut changed = false;
        for (i, m) in member.iter_mut().enumerate() {
            let (j, _) = closest(i, &centers);
            if *m != j {
                *m = j;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![Array1::<f64>::zeros(p.ncols()); centers.len()];
        let mut counts = vec![0usize; centers.len()];
        for (i, m) in member.iter().enumerate() {
            sums[*m] += &p.row(i);
            counts[*m] += 1;
        }
        for ((c, sum), count) in centers.iter_mut().zip(sums).zip(counts) {
            if count > 0 {
                *c = sum / count as f64;
            }
        }
    }
    (0..n).map(|i| closest(i, &centers).1).sum()
}

/// Native node `node`'s value in `P`'s run `trace`.
fn node_value<'a>(artifact: &Artifact, trace: &'a Trace, node: usize) -> Result<&'a Array2<f64>, String> {
    let at = artifact.place(node).ok_or_else(|| error(format!("P holds no node {node}")))?;
    trace.values.get(at).ok_or_else(|| error(format!("no value of node {at}")))
}

fn push_slice(part: &mut Dumped, operator: String, write: Vec<f64>, read: Vec<f64>) {
    let entry = part.slices.entry(operator).or_default();
    entry.0.push(write);
    entry.1.push(read);
}

fn set_gate(part: &mut Dumped, read: String, (on, tau, g): StageGate) {
    let kind = if g.is_some() { "direction" } else { "own" };
    part.gate = serde_json::json!({"kind": kind, "read": read, "tau": tau});
    part.g = g;
    part.on = on;
}

/// One arm's parts, in the layout of the toy gate's harness (`bench/toys_2951/score_toys.py`), at
/// `artifact`'s values (a checkpoint's posterior mean, or the start's): a part per gate with its
/// member components (each component of an unshared stage; components that share a gate, gate
/// sharing, one part of their summed rank), per part its slices' writes `U` and reads `V` on each of `M`'s operators it
/// spans (export names, `W = U Vᵀ`), its gate (own: on where `‖V_bᵀx‖ > τ_b` over its reads at
/// its stage, the gate `ln ‖V_bᵀx‖ − ln τ_b`; direction: `z = g_bᵀx − τ_b`, the threshold's constant folded into `τ_b`), and its hard
/// gate `z > 0` on every row of `inputs` in `P`'s own run (`artifact` executed on the host; a down
/// gate on its own read reads the MLP's activations recomputed from the gated `c_fc` reads, as the
/// MLP's rule computes them). Written to `dir`: `parts.json`, each slice set's `U` (out × r) and
/// `V` (in × r) and each direction's `g` as float64, and `active.f64` (rows × parts). Heads are
/// taken to own their k and v (no grouped-query sharing). Returns the number of parts.
pub fn dump_parts(native: &OperatorProgram, layers: &[LayerNodes], artifact: &Artifact, start: &Path, arm: &str, inputs: &FamilyInputs, dir: &Path) -> Result<usize, String> {
    let records: Vec<ArmRecord> = serde_json::from_slice(&std::fs::read(start).map_err(|e| error(format!("{}: {e}", start.display())))?).map_err(error)?;
    let record = records.into_iter().find(|r| r.arm == arm).ok_or_else(|| error(format!("{}: no arm {arm}", start.display())))?;
    if !record.means.is_empty() {
        return Err(error("dump_parts: mean parts are not dumped"));
    }
    let components = record.components;
    if components.iter().any(|c| matches!(c.read, Read::Active { .. } | Read::Logits { .. } | Read::Detectors { .. }) || c.score.is_some()) {
        return Err(error("dump_parts: all-on activation, logit, detector and score reads are not dumped"));
    }
    let read_site = |c: &Component| match &c.read {
        Read::Own([site, _]) => *site,
        Read::Direction { site, .. } | Read::Active { site } | Read::Logits { site } | Read::Detectors { site, .. } => *site,
    };
    // Per layer and stage, the components gated there, as `explanation` orders them.
    let mut at: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (b, c) in components.iter().enumerate() {
        let site = read_site(c);
        let home = (site / KINDS.len(), stage(site));
        if c.slices.iter().any(|&[s, _]| {
            let place = (s / KINDS.len(), stage(s));
            place != home && !(place.0 == home.0 && matches!((home.1, place.1), (0, 1) | (2, 3)))
        }) {
            return Err(error(format!("component {b}: dump_parts does not dump a block across blocks")));
        }
        at.entry((site / KINDS.len(), stage(site))).or_default().push(b);
    }
    let trace = artifact.execute(inputs)?;
    let rows = inputs.rows;
    let value = |node: usize| node_value(artifact, &trace, node);
    let matrix = |name: &str| -> Result<Array2<f64>, String> { Ok(artifact.program.operators[index_of(&artifact.program, name)?].matrix()) };
    let interfaces = native.interfaces().map_err(error)?;
    // A stage's gates, per component of the stage (its rows of the stacked `read`, `widths` of
    // them in order), from the stage's input `x`: in a shared stage each component takes the gate
    // its assignment puts it on (the largest entry of its column; `explanation` builds the gates'
    // pre-activations on the squared norm of their members' reads, `τ|τ|` the threshold), whose
    // `τ` in the norm's units it reports. And per gate with members its report ([`GateReport`]).
    let gates = |x: &Array2<f64>, read: &Array2<f64>, widths: &[usize], prefix: &str| -> Result<(Vec<StageGate>, Vec<GateReport>), String> {
        let threshold = matrix(&format!("{prefix}.threshold"))?;
        let assign = index_of(&artifact.program, &format!("{prefix}.assign")).ok().map(|op| artifact.program.operators[op].matrix());
        let directions = index_of(&artifact.program, &format!("{prefix}.direction")).ok().map(|op| artifact.program.operators[op].matrix());
        let count = widths.len();
        let gate_of: Vec<usize> = (0..count).map(|b| assign.as_ref().and_then(|a| a.column(b).iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|(m, _)| m)).unwrap_or(b)).collect();
        let a = x.dot(&read.t());
        let mut spans = Vec::with_capacity(count);
        let mut first = 0;
        for &w in widths {
            spans.push(first..first + w);
            first += w;
        }
        // The stage's own gates; a component assigned past them follows a component of the previous
        // stage (cross-stage sharing), which the caller fills in. Own gates pool their members'
        // reads only where the assignment is over the stage's own gates alone.
        let own = threshold.nrows();
        let pooled = assign.as_ref().is_some_and(|a| a.nrows() == own);
        let z: Vec<Array1<f64>> = match (&directions, &assign) {
            (Some(g), _) => (0..own).map(|m| x.dot(&g.row(m)) + threshold[[m, 0]]).collect(),
            (None, Some(assign)) if pooled => {
                let squares: Vec<Array1<f64>> = spans.iter().map(|r| a.slice(s![.., r.clone()]).map_axis(Axis(1), |row| row.dot(&row))).collect();
                (0..threshold.nrows())
                    .map(|m| {
                        let mut q = Array1::from_elem(x.nrows(), threshold[[m, 0]]);
                        for (b, square) in squares.iter().enumerate() {
                            q.scaled_add(assign[[m, b]], square);
                        }
                        q
                    })
                    .collect()
            }
            (None, _) => spans.iter().zip(threshold.column(0)).map(|(r, t)| a.slice(s![.., r.clone()]).map_axis(Axis(1), |row| row.dot(&row).sqrt().max(gam_gpu::tensor::LOG_FLOOR).ln()) + *t).collect(),
        };
        // An own gate's τ in its read norm's units: the pooled gate's on the squared norm, the
        // unpooled one's on its log.
        let tau = |m: usize| match (&directions, pooled) {
            (Some(_), _) => -threshold[[m, 0]],
            (None, true) => -threshold[[m, 0]].signum() * threshold[[m, 0]].abs().sqrt(),
            (None, false) => (-threshold[[m, 0]]).exp(),
        };
        let out = gate_of
            .iter()
            .map(|&m| if m < own { (z[m].iter().map(|v| *v > 0.0).collect(), tau(m), directions.as_ref().map(|g| g.row(m).to_vec())) } else { (vec![false; x.nrows()], f64::NAN, None) })
            .collect();
        let mut reports = Vec::new();
        for (m, zm) in z.iter().enumerate() {
            let members: Vec<usize> = (0..count).filter(|b| gate_of[*b] == m).collect();
            if members.is_empty() {
                continue;
            }
            let on: Vec<usize> = (0..x.nrows()).filter(|r| zm[*r] > 0.0).collect();
            let columns: Vec<usize> = members.iter().flat_map(|b| spans[*b].clone()).collect();
            let points = a.select(Axis(0), &on).select(Axis(1), &columns);
            reports.push((members, on.len() as f64 / x.nrows().max(1) as f64, contexts(&points)));
        }
        Ok((out, reports))
    };
    // Per gate with members: its stage, its members (indices into the arm's components), the
    // share of rows it is on and its contexts.
    let mut reported: Vec<(String, Vec<usize>, f64, usize)> = Vec::new();
    let mut parts: Vec<Dumped> = components.iter().map(|_| Dumped { slices: BTreeMap::new(), gate: serde_json::Value::Null, g: None, on: vec![false; rows] }).collect();
    for (l, layer) in layers.iter().enumerate() {
        let name = format!("library.l{l}");
        let site = |kind: Kind| KINDS.len() * l + KINDS.iter().position(|k| *k == kind).unwrap_or(0);
        let slices_on = |b: usize, s: usize| components[b].slices.iter().filter(|[t, _]| *t == s).count();
        let export = |s: usize| format!("blocks.{l}.{}", EXPORT_NAMES[s % KINDS.len()]);
        // ---------------------------------------------------------------- the attention's input stage
        let a_comps = at.get(&(l, 0)).cloned().unwrap_or_default();
        let (q, k, v) = (site(Kind::Query), site(Kind::Key), site(Kind::Value));
        // Per stacked row its component and site, as `explanation` stacks them.
        let mut read_rows: Vec<(usize, usize)> = Vec::new();
        let mut widths = Vec::new();
        for &b in &a_comps {
            let mut w = 0;
            for s in [q, k, v] {
                for _ in 0..slices_on(b, s) {
                    read_rows.push((b, s));
                    w += 1;
                }
            }
            widths.push(w);
        }
        if !a_comps.is_empty() {
            let read = matrix(&format!("{name}.attn.read"))?;
            let heads = layer.reads.len();
            let writes: Vec<Vec<Array2<f64>>> = (0..heads).map(|h| ["q", "k", "v"].iter().map(|p| matrix(&format!("{name}.h{h}.{p}"))).collect()).collect::<Result<_, _>>()?;
            // A row's write is its column among its map's rows, in every head.
            let mut column = [0usize; 3];
            for (r, &(b, s)) in read_rows.iter().enumerate() {
                let j = [q, k, v].iter().position(|t| *t == s).unwrap_or(0);
                let c = column[j];
                column[j] += 1;
                let u: Vec<f64> = (0..heads).flat_map(|h| writes[h][j].column(c).to_vec()).collect();
                push_slice(&mut parts[b], export(s), u, read.row(r).to_vec());
            }
            let (stage_gates, reports) = gates(value(layer.normed_stream)?, &read, &widths, &format!("{name}.attn"))?;
            reported.extend(reports.into_iter().map(|(members, on, k)| (format!("{name}.attn"), members.iter().map(|i| a_comps[*i]).collect(), on, k)));
            for (i, g) in stage_gates.into_iter().enumerate() {
                let b = a_comps[i];
                let first = read_rows.iter().find(|(c, _)| *c == b).map_or(q, |(_, s)| *s);
                set_gate(&mut parts[b], export(first), g);
            }
        }
        // ---------------------------------------------------------------- the attention's output
        let o = site(Kind::Output);
        let o_own = at.get(&(l, 1)).cloned().unwrap_or_default();
        let o_carriers: Vec<usize> = a_comps.iter().copied().filter(|&b| slices_on(b, o) > 0).chain(o_own.iter().copied()).collect();
        if !o_carriers.is_empty() {
            let (read, write) = (matrix(&format!("{name}.o.read"))?, matrix(&format!("{name}.o.write"))?);
            let (mut r, mut own_rows, mut own_widths) = (0, Vec::new(), Vec::new());
            for &b in &o_carriers {
                let n = slices_on(b, o);
                for _ in 0..n {
                    push_slice(&mut parts[b], export(o), write.column(r).to_vec(), read.row(r).to_vec());
                    if o_own.contains(&b) {
                        own_rows.push(r);
                    }
                    r += 1;
                }
                if o_own.contains(&b) {
                    own_widths.push(n);
                }
            }
            if !o_own.is_empty() {
                // The o gates on their own read read every head's output, concatenated.
                let heads: Vec<&Array2<f64>> = layer.reads.iter().map(|&n| value(n)).collect::<Result<_, _>>()?;
                let views: Vec<_> = heads.iter().map(|h| h.view()).collect();
                let concat = ndarray::concatenate(Axis(1), &views).map_err(error)?;
                let (stage_gates, reports) = gates(&concat, &read.select(Axis(0), &own_rows), &own_widths, &format!("{name}.o"))?;
                reported.extend(reports.into_iter().map(|(members, on, k)| (format!("{name}.o"), members.iter().map(|i| o_own[*i]).collect(), on, k)));
                for (i, g) in stage_gates.into_iter().enumerate() {
                    set_gate(&mut parts[o_own[i]], export(o), g);
                }
            }
        }
        // ---------------------------------------------------------------- the MLP
        let (fc, dn) = (site(Kind::Up), site(Kind::Down));
        let f_comps = at.get(&(l, 2)).cloned().unwrap_or_default();
        let d_own = at.get(&(l, 3)).cloned().unwrap_or_default();
        if f_comps.is_empty() {
            continue;
        }
        let (fc_read, fc_write) = (matrix(&format!("{name}.mlp.fc_read"))?, matrix(&format!("{name}.mlp.fc_write"))?);
        let (dn_read, dn_write) = (matrix(&format!("{name}.mlp.dn_read"))?, matrix(&format!("{name}.mlp.dn_write"))?);
        let (mut r, mut f_widths) = (0, Vec::new());
        for &b in &f_comps {
            let n = slices_on(b, fc);
            for _ in 0..n {
                push_slice(&mut parts[b], export(fc), fc_write.column(r).to_vec(), fc_read.row(r).to_vec());
                r += 1;
            }
            f_widths.push(n);
        }
        let h2 = value(layer.normed)?;
        let (f_gates, reports) = gates(h2, &fc_read, &f_widths, &format!("{name}.mlp.fc"))?;
        reported.extend(reports.into_iter().map(|(members, on, k)| (format!("{name}.mlp.fc"), members.iter().map(|i| f_comps[*i]).collect(), on, k)));
        let dn_carriers: Vec<usize> = f_comps.iter().copied().filter(|&b| slices_on(b, dn) > 0).chain(d_own.iter().copied()).collect();
        let (mut r, mut own_rows, mut own_widths) = (0, Vec::new(), Vec::new());
        for &b in &dn_carriers {
            let n = slices_on(b, dn);
            for _ in 0..n {
                push_slice(&mut parts[b], export(dn), dn_write.column(r).to_vec(), dn_read.row(r).to_vec());
                if d_own.contains(&b) {
                    own_rows.push(r);
                }
                r += 1;
            }
            if d_own.contains(&b) {
                own_widths.push(n);
            }
        }
        if !d_own.is_empty() {
            // The activations in P's run: the c_fc reads gated per component, written, through the
            // MLP's law.
            let mut gated = h2.dot(&fc_read.t());
            let mut first = 0;
            for (k, &w) in f_widths.iter().enumerate() {
                for (row, on) in f_gates[k].0.iter().enumerate() {
                    if !on {
                        gated.slice_mut(s![row, first..first + w]).fill(0.0);
                    }
                }
                first += w;
            }
            let mut act = gated.dot(&fc_write.t());
            let Node::Pointwise { input: pre, laws } = &native.nodes[layer.active] else {
                return Err(error(format!("layer {l}: the MLP activation is not one pointwise law")));
            };
            for (group, law) in laws.iter().enumerate() {
                let law = *law;
                act.slice_mut(s![.., interfaces[*pre].range(group)]).mapv_inplace(|t| law.apply(t));
            }
            let (stage_gates, reports) = gates(&act, &dn_read.select(Axis(0), &own_rows), &own_widths, &format!("{name}.mlp.dn"))?;
            reported.extend(reports.into_iter().map(|(members, on, k)| (format!("{name}.mlp.dn"), members.iter().map(|i| d_own[*i]).collect(), on, k)));
            // A down component assigned past the down gates follows a c_fc component: its gate,
            // and its part's (the c_fc component's gate's) membership.
            let follows: Vec<Option<usize>> = match index_of(&artifact.program, &format!("{name}.mlp.dn.assign")) {
                Ok(op) => {
                    let a = artifact.program.operators[op].matrix();
                    (0..d_own.len()).map(|b| a.column(b).iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|(m, _)| m).filter(|m| *m >= d_own.len()).map(|m| m - d_own.len())).collect()
                }
                Err(_) => vec![None; d_own.len()],
            };
            for (i, g) in stage_gates.into_iter().enumerate() {
                match follows[i] {
                    Some(c) => {
                        set_gate(&mut parts[d_own[i]], export(fc), f_gates[c].clone());
                        if let Some(part) = reported.iter_mut().find(|(stage, members, _, _)| *stage == format!("{name}.mlp.fc") && members.contains(&f_comps[c])) {
                            part.1.push(d_own[i]);
                        }
                    }
                    None => set_gate(&mut parts[d_own[i]], export(dn), g),
                }
            }
        }
        for (i, g) in f_gates.into_iter().enumerate() {
            set_gate(&mut parts[f_comps[i]], export(fc), g);
        }
    }
    std::fs::create_dir_all(dir).map_err(error)?;
    let write = |file: &str, values: &[f64]| -> Result<(), String> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(file), bytes).map_err(error)
    };
    // The parts written: one per gate with its member components (`reported`), so a stage's
    // components that came to share a gate (gate sharing) are one part of their summed rank, its
    // slices every member's, its gate and activity the gate's (every member's alike); an unshared
    // stage's parts are its components. A component no gate lists stays a part of its own.
    let mut listed = vec![false; parts.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (_, members, _, _) in &reported {
        if !members.is_empty() {
            members.iter().for_each(|b| listed[*b] = true);
            groups.push(members.clone());
        }
    }
    groups.extend((0..parts.len()).filter(|b| !listed[*b]).map(|b| vec![b]));
    let merged: Vec<Dumped> = groups
        .iter()
        .map(|members| {
            let mut slices: BTreeMap<String, (Vec<Vec<f64>>, Vec<Vec<f64>>)> = BTreeMap::new();
            for b in members {
                for (op, (us, vs)) in &parts[*b].slices {
                    let entry = slices.entry(op.clone()).or_default();
                    entry.0.extend(us.iter().cloned());
                    entry.1.extend(vs.iter().cloned());
                }
            }
            let first = &parts[members[0]];
            Dumped { slices, gate: first.gate.clone(), g: first.g.clone(), on: first.on.clone() }
        })
        .collect();
    let mut records = Vec::with_capacity(merged.len());
    for (b, (part, members)) in merged.iter().zip(&groups).enumerate() {
        let mut slices = serde_json::Map::new();
        for (op, (us, vs)) in &part.slices {
            // U as out × r and V as in × r, row-major.
            let (out, input) = (us[0].len(), vs[0].len());
            let u: Vec<f64> = (0..out).flat_map(|i| us.iter().map(move |c| c[i])).collect();
            let v: Vec<f64> = (0..input).flat_map(|i| vs.iter().map(move |c| c[i])).collect();
            write(&format!("p{b}.{op}.U.f64"), &u)?;
            write(&format!("p{b}.{op}.V.f64"), &v)?;
            slices.insert(op.clone(), serde_json::json!({"U": format!("p{b}.{op}.U.f64"), "V": format!("p{b}.{op}.V.f64"), "rank": us.len()}));
        }
        let mut gate = part.gate.clone();
        if let Some(g) = &part.g {
            write(&format!("p{b}.g.f64"), g)?;
            gate["g"] = serde_json::json!(format!("p{b}.g.f64"));
        }
        let name = if members.len() == 1 { format!("component {}", members[0]) } else { format!("part of components {members:?}") };
        records.push(serde_json::json!({"name": name, "slices": slices, "gate": gate, "members": members}));
    }
    let active: Vec<f64> = (0..rows).flat_map(|row| merged.iter().map(move |p| if p.on[row] { 1.0 } else { 0.0 })).collect();
    write("active.f64", &active)?;
    let record = serde_json::json!({"parts": records, "active": "active.f64", "kept": ["wte", "lm_head"], "fitter": format!("library_vpd, arm {arm}")});
    std::fs::write(dir.join("parts.json"), serde_json::to_vec_pretty(&record).map_err(error)?).map_err(error)?;
    // Per part (a gate with its member components, of any rank where a stage shares gates): its
    // rank (the slices its members run, in rank-one equivalents), the share of rows it is on and
    // the distinct contexts it fires in; per stage the parts' rank histogram.
    let rank_of = |b: usize| parts[b].slices.values().map(|(us, _)| us.len()).sum::<usize>();
    let report: Vec<serde_json::Value> = reported
        .iter()
        .map(|(stage, members, on, k)| serde_json::json!({"stage": stage, "members": members, "rank": members.iter().map(|b| rank_of(*b)).sum::<usize>(), "share_on": on, "contexts": k}))
        .collect();
    let mut stages: BTreeMap<&str, BTreeMap<usize, usize>> = BTreeMap::new();
    for (stage, members, _, _) in &reported {
        *stages.entry(stage.as_str()).or_default().entry(members.iter().map(|b| rank_of(*b)).sum()).or_insert(0) += 1;
    }
    let summary = serde_json::json!({"parts": report, "rank_histogram": stages, "rows": rows});
    std::fs::write(dir.join("parts_report.json"), serde_json::to_vec_pretty(&summary).map_err(error)?).map_err(error)?;
    let parts = merged;
    Ok(parts.len())
}

#[cfg(test)]
mod tests {
    use crate::{
        import::import_language_model,
        interchange::{Batch, Experiment, Interchange},
        operator_program::SlotValues,
        run_check::{layer_nodes, split_sites},
    };
    use gam_gpu::tensor::Device;
    use ndarray::{Array1, Array2};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// `A⁻¹` of a small square matrix by Gauss-Jordan elimination with partial pivoting.
    fn inverse(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let mut m = ndarray::concatenate![ndarray::Axis(1), a.clone(), Array2::eye(n)];
        for c in 0..n {
            let p = (c..n).max_by(|&i, &j| m[[i, c]].abs().total_cmp(&m[[j, c]].abs())).expect("a pivot row");
            for k in 0..2 * n {
                m.swap([c, k], [p, k]);
            }
            let pivot = m[[c, c]];
            m.row_mut(c).mapv_inplace(|v| v / pivot);
            for r in (0..n).filter(|&r| r != c) {
                let f = m[[r, c]];
                let row = m.row(c).to_owned();
                m.row_mut(r).scaled_add(-f, &row);
            }
        }
        m.slice(ndarray::s![.., n..]).to_owned()
    }

    /// Exact dense random frames of every map of the tiny export `dir` (2 layers), written to
    /// `factors` in VPD's export layout: per site its frame `(V, U)` and its input width.
    fn exact_frames(dir: &std::path::Path, factors: &std::path::Path, seed: u64) -> Vec<(Array2<f64>, Array2<f64>, usize)> {
        let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("export.json")).expect("export.json")).expect("the record");
        std::fs::create_dir_all(factors).expect("the factors' directory");
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut files, mut sites, mut out) = (serde_json::Map::new(), Vec::new(), Vec::new());
        for l in 0..2 {
            for name in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"] {
                let shape = &record["files"][format!("blocks.{l}.{name}")]["shape"];
                let (r, c) = (shape[0].as_u64().expect("rows") as usize, shape[1].as_u64().expect("columns") as usize);
                let w = crate::import::read_f64_shaped(&dir.join(format!("blocks.{l}.{name}.f64")), r, c).expect("a map");
                let v = Array2::from_shape_fn((c, c + 4), |_| rng.random::<f64>() - 0.5);
                let u = inverse(&v.dot(&v.t())).dot(&v).t().dot(&w.t());
                assert!((v.dot(&u) - w.t()).iter().all(|e| e.abs() < 1e-9), "an exact frame");
                let site = format!("h.{l}.{name}");
                for (part, values) in [("U", &u), ("V", &v)] {
                    let bytes: Vec<u8> = values.iter().flat_map(|e| e.to_le_bytes()).collect();
                    std::fs::write(factors.join(format!("{site}.{part}.f64")), bytes).expect("a factor");
                    files.insert(format!("{site}.{part}"), serde_json::json!({"shape": [values.nrows(), values.ncols()]}));
                }
                sites.push(site);
                out.push((v, u, c));
            }
        }
        std::fs::write(factors.join("export.json"), serde_json::json!({"config": {"sites": sites}, "files": files}).to_string()).expect("the factors' record");
        out
    }

    /// Dense random frames of every map of the tiny export, exact (`V` [in × C] random with
    /// `C = in + 4` slices, `U = Vᵀ(VVᵀ)⁻¹Wᵀ`, so `V U = Wᵀ` with no identity structure), every
    /// component on, under each kind of gate read: own reads at every stage; directions at every
    /// stage (a small random `g`, a constant far above it); stages of both kinds in one
    /// explanation (own attention and down gates, direction c_fc gates, as a neuron start's export
    /// beside its always-on attention); and c_fc-and-down components gated on their down reads of the
    /// all-on activations (`Read::Active`). Each arm's `P` computes `M` (`KL(M ‖ P)` under 10⁻⁵ bits
    /// over every token), which a gate mode chosen for the whole explanation broke (an own gate in
    /// a direction explanation read the constant 0, `Φ(0) = 1/2`, halving its slices). A stage
    /// with both kinds of read is refused.
    #[test]
    fn dense_frames_all_on_compute_m_under_every_gate_read() {
        let tag = "library_vpd_dense";
        let dir = crate::test_support::tiny_export(tag, 2);
        let factors = std::env::temp_dir().join(format!("gam_mpd_{tag}_factors_{}", std::process::id()));
        let frames = exact_frames(&dir, &factors, 5);
        let (count, inputs): (Vec<usize>, Vec<usize>) = frames.iter().map(|(v, _, c)| (v.ncols(), *c)).unzip();
        let mut rng = StdRng::seed_from_u64(6);
        // Component reads: own (threshold −1 below a norm) or a direction (g small, c = 10).
        let mut direction = |site: usize| serde_json::json!({"direction": {"site": site, "coefficients": (0..=inputs[site]).map(|j| if j < inputs[site] { 0.01 * (rng.random::<f64>() - 0.5) } else { 10.0 }).collect::<Vec<f64>>()}});
        let mut arms = Vec::new();
        for (arm, [attention, up, down]) in [("own", [false; 3]), ("direction", [true; 3]), ("stages", [false, true, false]), ("mixed", [false, true, true]), ("active", [false; 3]), ("active_gated", [false; 3]), ("active_shared", [false; 3]), ("own_gated", [false; 3]), ("detectors", [false; 3]), ("detectors_gated", [false; 3])] {
            let mut components = Vec::new();
            for l in 0..2 {
                let (q, o, fc, dn) = (6 * l, 6 * l + 3, 6 * l + 4, 6 * l + 5);
                let read = |own: bool, site: usize, direction: &mut dyn FnMut(usize) -> serde_json::Value| if own { serde_json::json!({"own": [site, 0]}) } else { direction(site) };
                let slices: Vec<[usize; 2]> = (q..=o).flat_map(|s| (0..count[s]).map(move |i| [s, i])).collect();
                components.push(serde_json::json!({"read": read(!attention, q, &mut direction), "tau": -1.0, "slices": slices}));
                // c_fc slice i with down slice i (a neuron-like part), the down slices past c_fc's
                // on their own down gates; "mixed" puts one c_fc part on its own read.
                for i in 0..count[fc] {
                    let own = !up || (arm == "mixed" && i == 0);
                    let r = if arm.starts_with("active") {
                        serde_json::json!({"active": {"site": fc}})
                    } else if arm.starts_with("detectors") {
                        serde_json::json!({"detectors": {"site": fc, "terms": [[i, 1.0]]}})
                    } else if own {
                        serde_json::json!({"own": [fc, i]})
                    } else {
                        direction(fc)
                    };
                    // "active_gated", "active_shared", "own_gated" and "detectors_gated": thresholds
                    // that turn about half of the parts off; "active_shared" lets each move to the
                    // next part's gate.
                    let tau = if ["active_gated", "active_shared", "own_gated", "detectors_gated"].contains(&arm) { 0.05 } else { -1.0 };
                    let candidates: Vec<usize> = if arm == "active_shared" { vec![components.len() - i + (i + 1) % count[fc]] } else { Vec::new() };
                    components.push(serde_json::json!({"read": r, "tau": tau, "slices": [[fc, i], [dn, i]], "candidates": candidates}));
                }
                for i in count[fc]..count[dn] {
                    let r = if down { direction(dn) } else { serde_json::json!({"own": [dn, i]}) };
                    components.push(serde_json::json!({"read": r, "tau": -1.0, "slices": [[dn, i]]}));
                }
            }
            // A detector arm's MLP input stages: one detector per c_fc slice, its own read, at
            // threshold 0, each part's gate that detector alone (the own gate's form).
            let detectors: Vec<serde_json::Value> = if arm.starts_with("detectors") {
                (0..2)
                    .map(|l| {
                        let (v, _, _) = &frames[6 * l + 4];
                        let directions: Vec<Vec<f64>> = (0..v.ncols()).map(|i| v.column(i).to_vec()).collect();
                        serde_json::json!({"layer": l, "stage": 2, "thresholds": vec![0.0; directions.len()], "directions": directions})
                    })
                    .collect()
            } else {
                Vec::new()
            };
            arms.push(serde_json::json!({"arm": arm, "components": components, "detectors": detectors}));
        }
        let start = factors.join("start.json");
        std::fs::write(&start, serde_json::Value::Array(arms).to_string()).expect("the start");
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        let device = Device::host();
        for arm in ["own", "direction", "stages", "active", "detectors"] {
            let explanation = super::explanation(&native, &layers, &factors, &start, arm).expect("the explanation");
            let blocks: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
            let ic = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments");
            let clean: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 }).collect();
            let bits: f64 = ic.evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum();
            assert!(bits.abs() < 1e-5, "arm {arm}: KL(M ‖ P) = {bits} bits with every component on");
        }
        // Gated on the all-on activations, under neuron-group edits: the training path (both uses of
        // the c_fc map take its rows' edit; the gate's reads are its own, not the down map's) equals
        // M and P mutated directly (c_fc rows scaled in M's map and in P's slices' writes, down
        // columns in M's map and in P's slices' reads).
        let explanation = super::explanation(&native, &layers, &factors, &start, "active_gated").expect("the explanation");
        let blocks: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let edits: Vec<crate::weight_edit::Drawn> = crate::weight_edit::candidates(&native, 3, 40).expect("the edits").into_iter().filter(|d| d.kind == crate::weight_edit::Kind::Neurons).take(4).collect();
        assert!(!edits.is_empty(), "neuron-group edits drawn");
        let mut ic = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments");
        ic.set_weight_edits(edits.clone()).expect("the table");
        let clean: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 }).collect();
        let open: f64 = ic.evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum();
        assert!(open > 1e-3, "some parts off: KL(M ‖ P) = {open} bits");
        let scale = |program: &mut crate::operator_program::OperatorProgram, name: &str, rows: bool, units: &[usize], alpha: f64| {
            let op = super::index_of(program, name).expect("the operator");
            let old = std::sync::Arc::clone(&program.operators[op]);
            let mut v = old.matrix();
            for &u in units {
                if rows { v.row_mut(u).mapv_inplace(|x| x * alpha) } else { v.column_mut(u).mapv_inplace(|x| x * alpha) }
            }
            let precision = crate::operator_program::exact_precision(v.iter().copied()).expect("a precision");
            program.operators[op] = std::sync::Arc::new(crate::operator_program::Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), v, precision, old.provenance.clone()).expect("an operator"));
        };
        for (i, d) in edits.iter().enumerate() {
            let experiments: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: Some(crate::interchange::Patch::Weights { edit: i, block: d.block }), position: 0 }).collect();
            let training: f64 = ic.evaluate(&batch, &experiments, false).expect("evaluate").bits.iter().flatten().sum();
            let (mut m, mut p) = (native.clone(), explanation.artifact.clone());
            for e in &d.entries {
                scale(&mut m, &e.native, e.rows, &e.units, e.alpha);
                let l = e.native.split('.').nth(1).expect("a layer");
                let (part, rows) = if e.native.ends_with("c_fc") { ("fc_write", true) } else { ("dn_read", false) };
                scale(&mut p.program, &format!("library.l{l}.mlp.{part}"), rows, &e.units, e.alpha);
            }
            let direct: f64 = Interchange::new(&device, &m, &blocks, &p, &explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments").evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum();
            assert!((training - direct).abs() <= 1e-9 * direct.abs().max(1.0), "edit {i}: training {training} bits, direct mutation {direct}");
        }
        // Shared gates on the all-on activations: at the identity assignment each gate is its own
        // component's (`‖·‖² > τ²` where `‖·‖ > τ`), so the hard explanation scores as the unshared.
        let shared = super::explanation(&native, &layers, &factors, &start, "active_shared").expect("the explanation");
        assert!(shared.shares.len() == 2, "one shared stage per layer");
        let bits: f64 = Interchange::new(&device, &native, &blocks, &shared.artifact, &shared.trainable, Vec::new(), 1 << 30, 64).expect("the experiments").evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum();
        assert!((bits - open).abs() <= 1e-9 * open.max(1.0), "shared all-on gates {bits} bits against unshared {open}");
        // Detector gates with one detector each, the part's own read at threshold 0, are its own
        // gates: the hard explanation scores as the own-gated one, some parts off.
        let score = |arm: &str| -> f64 {
            let explanation = super::explanation(&native, &layers, &factors, &start, arm).expect("the explanation");
            Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments").evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum()
        };
        let (own, detected) = (score("own_gated"), score("detectors_gated"));
        assert!(own > 1e-3 && (own - detected).abs() <= 1e-9 * own, "detector gates {detected} bits against own gates {own}");
        let mixed = super::explanation(&native, &layers, &factors, &start, "mixed");
        assert!(mixed.as_ref().is_err_and(|e| e.contains("gates one way")), "a stage with both kinds of read is refused: {:?}", mixed.err());
        std::fs::remove_dir_all(&factors).expect("the factors are removed");
    }

    /// Mean parts (`LayerMeans`, from `M`'s run on the text itself): with every part on the
    /// explanation is `M` (under 1e-5 bits); with every part off each MLP writes its mean output
    /// `W_dn E[act]` on every token and its pre-activations are `E[p] = W_fc E[x]`, each head's v
    /// is its mean `W_v E[x]`, and the attention's output is `Σ_h W_o^h E[v_h]`: every part
    /// mean-ablated.
    #[test]
    fn mean_parts_keep_m_and_turn_off_to_the_mean() {
        use crate::operator_program::Node;
        let tag = "library_vpd_means";
        let dir = crate::test_support::tiny_export(tag, 2);
        let factors = std::env::temp_dir().join(format!("gam_mpd_{tag}_factors_{}", std::process::id()));
        let frames = exact_frames(&dir, &factors, 13);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let count = |site: usize| frames[site].0.ncols();
        let trace = native.execute(&imported.family, false).expect("M's trace");
        let mean = |node: usize| trace.values[node].mean_axis(ndarray::Axis(0)).expect("rows").to_vec();
        let means: Vec<serde_json::Value> = (0..2).map(|l| serde_json::json!({"layer": l, "attention": mean(layers[l].normed_stream), "mlp": mean(layers[l].normed), "activation": mean(layers[l].active)})).collect();
        let arm = |name: &str, tau: f64| {
            let mut components = Vec::new();
            for l in 0..2 {
                let (q, o, fc, dn) = (6 * l, 6 * l + 3, 6 * l + 4, 6 * l + 5);
                let slices: Vec<[usize; 2]> = (q..=o).flat_map(|s| (0..count(s)).map(move |i| [s, i])).collect();
                components.push(serde_json::json!({"read": {"own": [q, 0]}, "tau": tau, "slices": slices}));
                components.extend((0..count(fc)).map(|i| serde_json::json!({"read": {"own": [fc, i]}, "tau": tau, "slices": [[fc, i], [dn, i]]})));
                components.extend((count(fc)..count(dn)).map(|i| serde_json::json!({"read": {"own": [dn, i]}, "tau": tau, "slices": [[dn, i]]})));
            }
            serde_json::json!({"arm": name, "components": components, "means": means})
        };
        let start = factors.join("start.json");
        std::fs::write(&start, serde_json::Value::Array(vec![arm("on", -1.0), arm("off", 1e9)]).to_string()).expect("the start");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        let device = Device::host();
        let clean: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 }).collect();
        let on = super::explanation(&native, &layers, &factors, &start, "on").expect("the explanation");
        let blocks: Vec<_> = on.layers.iter().map(|l| l.sites.clone()).collect();
        let bits: f64 = Interchange::new(&device, &native, &blocks, &on.artifact, &on.trainable, Vec::new(), 1 << 30, 64).expect("the experiments").evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum();
        assert!(bits.abs() < 1e-5, "every part on with mean parts: KL(M ‖ P) = {bits} bits");
        let off = super::explanation(&native, &layers, &factors, &start, "off").expect("the explanation");
        let (flat, _, _) = crate::interchange::sites(&off.artifact, &layers).expect("the flat program");
        let p_trace = flat.execute(&imported.family, false).expect("P's trace");
        for l in 0..2 {
            let heads = layers[l].reads.len();
            let mut parts: Vec<String> = (0..heads).map(|h| format!("h{h}.v_mean")).collect();
            parts.extend(["o.mean".to_string(), "mlp.fc_mean".to_string()]);
            for part in parts {
                let bias = super::index_of(&flat, &format!("library.l{l}.{part}")).expect("the mean part");
                let node = flat.nodes.iter().position(|n| matches!(n, Node::Affine { bias: Some(b), .. } if *b == bias)).expect("its node");
                let row = flat.operators[bias].matrix().column(0).to_owned();
                let gap = p_trace.values[node].rows().into_iter().map(|r| (&r - &row).iter().fold(0.0_f64, |m, e| m.max(e.abs()))).fold(0.0_f64, f64::max);
                assert!(gap <= 1e-12 * row.iter().fold(1.0_f64, |m, e| m.max(e.abs())), "layer {l} {part}: every token writes the mean, off by {gap}");
            }
            let ex = Array1::from(mean(layers[l].normed_stream));
            let Node::Affine { terms, .. } = &native.nodes[layers[l].attention] else { panic!("the o map") };
            let mut want = Array1::<f64>::zeros(native.operators[terms[0].1].rows.width());
            for &read in &layers[l].reads {
                let Node::Attend { value, .. } = &native.nodes[read] else { panic!("a head") };
                let Node::Affine { terms: v, .. } = &native.nodes[*value] else { panic!("a value map") };
                let (_, op) = terms.iter().find(|(a, _)| *a == read).expect("the head's o columns");
                want += &native.operators[*op].matrix().dot(&native.operators[v[0].1].matrix().dot(&ex));
            }
            let o_mean = flat.operators[super::index_of(&flat, &format!("library.l{l}.o.mean")).expect("the o mean")].matrix().column(0).to_owned();
            let err = (&o_mean - &want).iter().fold(0.0_f64, |m, e| m.max(e.abs()));
            assert!(err <= 1e-12 * want.iter().fold(1.0_f64, |m, e| m.max(e.abs())), "layer {l}: the o mean part is Σ W_o^h E[v_h], off by {err}");
            for (part, want) in [("fc_mean", None), ("dn_mean", Some(()))] {
                let bias = super::index_of(&flat, &format!("library.l{l}.mlp.{part}")).expect("the mean part");
                let node = flat.nodes.iter().position(|n| matches!(n, Node::Affine { bias: Some(b), .. } if *b == bias)).expect("its node");
                let value = &p_trace.values[node];
                let row = flat.operators[bias].matrix().column(0).to_owned();
                let gap = value.rows().into_iter().map(|r| (&r - &row).iter().fold(0.0_f64, |m, e| m.max(e.abs()))).fold(0.0_f64, f64::max);
                assert!(gap <= 1e-12 * row.iter().fold(1.0_f64, |m, e| m.max(e.abs())), "layer {l} {part}: every token writes the mean, off by {gap}");
                if want.is_some() {
                    let w_dn = native.operators[super::mlp_down(&native, &layers[l]).expect("the down map")].matrix();
                    let expected = w_dn.dot(&Array1::from(mean(layers[l].active)));
                    let err = (&row - &expected).iter().fold(0.0_f64, |m, e| m.max(e.abs()));
                    assert!(err <= 1e-12 * expected.iter().fold(1.0_f64, |m, e| m.max(e.abs())), "layer {l}: the mean write is W_dn E[act], off by {err}");
                }
            }
        }
        std::fs::remove_dir_all(&factors).expect("the factors are removed");
    }

    /// Score gates (`ScoreRead`): q slices in threes on their own reads beside their head's largest
    /// score, each k slice and the v slices on their own reads, then the o slices and the MLP's.
    /// With every gate open the explanation is `M` (under 1e-5 bits); each q component's gate
    /// pre-activation equals its definition from `M`'s maps, `ln ‖V_bᵀx_t‖ + w max_{u ≤ t} c ⟨q̃_t, k̃_u⟩
    /// − ln τ_b` (its head's queries and keys turned to their positions), within 1e-8, and the
    /// scores move the gates (two tokens of a component differ in their score term).
    #[test]
    fn score_reads_gate_on_their_heads_largest_scores() {
        use crate::operator_program::Node;
        let tag = "library_vpd_scores";
        let dir = crate::test_support::tiny_export(tag, 2);
        let factors = std::env::temp_dir().join(format!("gam_mpd_{tag}_factors_{}", std::process::id()));
        let frames = exact_frames(&dir, &factors, 11);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let count = |site: usize| frames[site].0.ncols();
        let weight = 0.5;
        let mut components = Vec::new();
        for l in 0..2 {
            let heads = layers[l].reads.len();
            let (q, k, v, o, fc, dn) = (6 * l, 6 * l + 1, 6 * l + 2, 6 * l + 3, 6 * l + 4, 6 * l + 5);
            for (n, i) in (0..count(q)).step_by(3).enumerate() {
                let slices: Vec<[usize; 2]> = (i..(i + 3).min(count(q))).map(|j| [q, j]).collect();
                components.push(serde_json::json!({"read": {"own": [q, i]}, "tau": -1.0, "slices": slices, "score": {"head": n % heads, "weight": weight}}));
            }
            components.extend((0..count(k)).map(|i| serde_json::json!({"read": {"own": [k, i]}, "tau": -1.0, "slices": [[k, i]]})));
            components.push(serde_json::json!({"read": {"own": [v, 0]}, "tau": -1.0, "slices": (0..count(v)).map(|i| [v, i]).collect::<Vec<_>>()}));
            components.extend((0..count(o)).map(|i| serde_json::json!({"read": {"own": [o, i]}, "tau": -1.0, "slices": [[o, i]]})));
            components.extend((0..count(fc)).map(|i| serde_json::json!({"read": {"own": [fc, i]}, "tau": -1.0, "slices": [[fc, i], [dn, i]]})));
            components.extend((count(fc)..count(dn)).map(|i| serde_json::json!({"read": {"own": [dn, i]}, "tau": -1.0, "slices": [[dn, i]]})));
        }
        // The same arm with mean parts (`LayerMeans`, from `M`'s run on the text).
        let trace = native.execute(&imported.family, false).expect("M's trace");
        let mean = |node: usize| trace.values[node].mean_axis(ndarray::Axis(0)).expect("rows").to_vec();
        let means: Vec<serde_json::Value> = (0..2).map(|l| serde_json::json!({"layer": l, "attention": mean(layers[l].normed_stream), "mlp": mean(layers[l].normed), "activation": mean(layers[l].active)})).collect();
        let start = factors.join("start.json");
        std::fs::write(&start, serde_json::json!([{"arm": "scores", "components": components}, {"arm": "scores_means", "components": components, "means": means}]).to_string()).expect("the start");
        let explanation = super::explanation(&native, &layers, &factors, &start, "scores").expect("the explanation");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        let device = Device::host();
        let clean: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 }).collect();
        let blocks: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let bits: f64 = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments").evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum();
        assert!(bits.abs() < 1e-5, "every gate open: KL(M ‖ P) = {bits} bits");
        let centered = super::explanation(&native, &layers, &factors, &start, "scores_means").expect("the explanation");
        let bits: f64 = Interchange::new(&device, &native, &blocks, &centered.artifact, &centered.trainable, Vec::new(), 1 << 30, 64).expect("the experiments").evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum();
        assert!(bits.abs() < 1e-5, "every gate open with mean parts: KL(M ‖ P) = {bits} bits");
        let (flat, _, reads) = crate::interchange::sites(&explanation.artifact, &layers).expect("the flat program");
        let trace = flat.execute(&imported.family, false).expect("the trace");
        let layout = imported.family.layout.as_ref().expect("a layout");
        for l in 0..2 {
            let threshold = super::index_of(&flat, &format!("library.l{l}.attn.threshold")).expect("the thresholds");
            let gate = flat.nodes.iter().position(|n| matches!(n, Node::Affine { terms, bias: Some(b) } if *b == threshold && terms.len() == 1)).expect("the gate node");
            let x = &trace.values[reads[2 * l]];
            let op = |name: &str| super::index_of(&native, name).map(|i| native.operators[i].matrix()).expect("a map");
            let Node::Attend { scale, rotary, .. } = native.nodes[layers[l].reads[0]].clone() else { panic!("an attend node") };
            let c = scale.value();
            let heads = layers[l].reads.len();
            let turned = |v: Vec<f64>, p: u32| {
                let mut v = v;
                if let Some(r) = rotary {
                    r.rotate(&mut v, None, p);
                }
                Array1::from(v)
            };
            let rows = x.nrows();
            let largest = Array2::from_shape_fn((rows, heads), |(t, h)| {
                let (qm, km) = (x.dot(&op(&format!("blocks.{l}.q{h}")).t()), x.dot(&op(&format!("blocks.{l}.k{h}")).t()));
                let qt = turned(qm.row(t).to_vec(), layout.position[t]);
                (0..rows)
                    .filter(|&u| layout.sequence[u] == layout.sequence[t] && layout.position[u] <= layout.position[t])
                    .map(|u| c * qt.dot(&turned(km.row(u).to_vec(), layout.position[u])))
                    .fold(f64::NEG_INFINITY, f64::max)
            });
            let q = 6 * l;
            let z = &trace.values[gate];
            let mut moved = false;
            for (n, i) in (0..count(q)).step_by(3).enumerate() {
                let h = n % heads;
                for t in 0..rows {
                    let norm = (i..(i + 3).min(count(q))).map(|j| x.row(t).dot(&frames[q].0.column(j)).powi(2)).sum::<f64>().sqrt();
                    let want = norm.max(gam_gpu::tensor::LOG_FLOOR).ln() + weight * largest[[t, h]] + super::log_threshold(-1.0);
                    assert!((z[[t, n]] - want).abs() <= 1e-8 * want.abs().max(1.0), "layer {l} component {n} token {t}: {} against {want}", z[[t, n]]);
                }
                moved |= (largest[[0, h]] - largest[[rows - 1, h]]).abs() > 1e-6;
            }
            assert!(moved, "layer {l}: the scores move the gates");
        }
        std::fs::remove_dir_all(&factors).expect("the factors are removed");
    }

    /// Query components gated on their logits' spread (`Read::Logits`) beside own-read k, v, o and
    /// MLP components, on exact dense frames of the tiny export: with every gate open `P` is `M`
    /// (under 10⁻⁵ bits); in each layer every attention input component's gate input is its
    /// definition computed here from `M`'s maps (each head's q and k on the layer's input, the
    /// causal pattern, the keys turned to their positions, their covariance turned to the query's
    /// frame by its position's rotation, its diagonal weighted by the slices' squared writes and
    /// the scale's square; an own read's squared norm) within 10⁻⁸, before and after gates at
    /// their medians shut some components; and head edits (q, k, v rows, o columns) score through
    /// the training path as `M` and `P` mutated directly within 10⁻⁹ bits.
    #[test]
    fn logit_reads_gate_on_the_spread_of_their_logits() {
        use crate::operator_program::{Node, OperatorProgram};
        let tag = "library_vpd_logits";
        let dir = crate::test_support::tiny_export(tag, 2);
        let factors = std::env::temp_dir().join(format!("gam_mpd_{tag}_factors_{}", std::process::id()));
        let frames = exact_frames(&dir, &factors, 9);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let count = |site: usize| frames[site].0.ncols();
        // Per layer the attention input stage's components: q slices in threes on logit reads,
        // each k slice on its own read, the v slices together; then the o slices and the MLP's.
        let arm = |name: &str, taus: &[Vec<f64>]| {
            let mut components = Vec::new();
            for l in 0..2 {
                let (q, k, v, o, fc, dn) = (6 * l, 6 * l + 1, 6 * l + 2, 6 * l + 3, 6 * l + 4, 6 * l + 5);
                let mut stage: Vec<(serde_json::Value, Vec<[usize; 2]>)> = (0..count(q)).step_by(3).map(|i| (serde_json::json!({"logits": {"site": q}}), (i..(i + 3).min(count(q))).map(|j| [q, j]).collect())).collect();
                stage.extend((0..count(k)).map(|i| (serde_json::json!({"own": [k, i]}), vec![[k, i]])));
                stage.push((serde_json::json!({"own": [v, 0]}), (0..count(v)).map(|i| [v, i]).collect()));
                let last = stage.len() - 1;
                for (b, (read, slices)) in stage.into_iter().enumerate() {
                    // The v slices always on at an export's threshold (export_to_rust.py's −10⁹).
                    let tau = if b == last && !taus.is_empty() { -1e9 } else { taus.get(l).map_or(-1.0, |t| t[b]) };
                    components.push(serde_json::json!({"read": read, "tau": tau, "slices": slices}));
                }
                for i in 0..count(o) {
                    components.push(serde_json::json!({"read": {"own": [o, i]}, "tau": -1.0, "slices": [[o, i]]}));
                }
                for i in 0..count(fc) {
                    components.push(serde_json::json!({"read": {"own": [fc, i]}, "tau": -1.0, "slices": [[fc, i], [dn, i]]}));
                }
                for i in count(fc)..count(dn) {
                    components.push(serde_json::json!({"read": {"own": [dn, i]}, "tau": -1.0, "slices": [[dn, i]]}));
                }
            }
            serde_json::json!({"arm": name, "components": components})
        };
        let start = factors.join("start.json");
        let build = |name: &str, taus: &[Vec<f64>]| {
            std::fs::write(&start, serde_json::Value::Array(vec![arm(name, taus)]).to_string()).expect("the start");
            super::explanation(&native, &layers, &factors, &start, name).expect("the explanation")
        };
        // Per layer, each stage component's gate input in the flat program on every token, and
        // the definition's from the layer's input there.
        let gate_inputs = |explanation: &crate::library_mdl::Explanation| -> Vec<(Array2<f64>, Array2<f64>)> {
            let (flat, _, reads) = crate::interchange::sites(&explanation.artifact, &layers).expect("the flat program");
            let trace = flat.execute(&imported.family, false).expect("the trace");
            let layout = imported.family.layout.as_ref().expect("a layout");
            (0..2)
                .map(|l| {
                    let threshold = super::index_of(&flat, &format!("library.l{l}.attn.threshold")).expect("the thresholds");
                    let sum = flat.nodes.iter().find_map(|n| match n {
                        Node::Affine { terms, bias: Some(b) } if *b == threshold => Some(terms[0].0),
                        _ => None,
                    }).expect("the gate node");
                    let x = &trace.values[reads[2 * l]];
                    let op = |name: &str| super::index_of(&native, name).map(|i| native.operators[i].matrix()).expect("a map");
                    let Node::Attend { scale, rotary, .. } = native.nodes[layers[l].reads[0]].clone() else { panic!("an attend node") };
                    let (c, rotary) = (scale.value(), rotary.expect("a rotation"));
                    let heads = layers[l].reads.len();
                    let hd = op(&format!("blocks.{l}.q0")).nrows();
                    let turned = |v: &[f64], p: u32| {
                        let mut v = v.to_vec();
                        rotary.rotate(&mut v, None, p);
                        Array1::from(v)
                    };
                    // Per row and head the keys' coordinate variances in the query's frame.
                    let rows = x.nrows();
                    let mut spread = Array2::<f64>::zeros((rows, heads * hd));
                    for h in 0..heads {
                        let (qm, km) = (x.dot(&op(&format!("blocks.{l}.q{h}")).t()), x.dot(&op(&format!("blocks.{l}.k{h}")).t()));
                        for t in 0..rows {
                            let keys: Vec<usize> = (0..rows).filter(|&s| layout.sequence[s] == layout.sequence[t] && layout.position[s] <= layout.position[t]).collect();
                            let qt = turned(qm.row(t).as_slice().expect("a row"), layout.position[t]);
                            let kt: Vec<Array1<f64>> = keys.iter().map(|&s| turned(km.row(s).to_vec().as_slice(), layout.position[s])).collect();
                            let scores: Vec<f64> = kt.iter().map(|k| c * qt.dot(k)).collect();
                            let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            let e: Vec<f64> = scores.iter().map(|z| (z - top).exp()).collect();
                            let total: f64 = e.iter().sum();
                            let mean = kt.iter().zip(&e).fold(Array1::<f64>::zeros(hd), |m, (k, w)| m + k * (w / total));
                            let cov = kt.iter().zip(&e).fold(Array2::<f64>::zeros((hd, hd)), |m, (k, w)| {
                                let d = k - &mean;
                                m + &(d.view().insert_axis(ndarray::Axis(1)).dot(&d.view().insert_axis(ndarray::Axis(0))) * (w / total))
                            });
                            for j in 0..hd {
                                let mut unit = vec![0.0; hd];
                                unit[j] = 1.0;
                                let r = turned(&unit, layout.position[t]);
                                spread[[t, h * hd + j]] = r.dot(&cov.dot(&r));
                            }
                        }
                    }
                    // Each component's definition: Σ_i c_i² Σ_k u_ik² D_k c² on a logit read, Σ_i c_i² on an own read.
                    let (q, k, v) = (6 * l, 6 * l + 1, 6 * l + 2);
                    let mut stage: Vec<Vec<(usize, usize, bool)>> = (0..count(q)).step_by(3).map(|i| (i..(i + 3).min(count(q))).map(|j| (q, j, true)).collect()).collect();
                    stage.extend((0..count(k)).map(|i| vec![(k, i, false)]));
                    stage.push((0..count(v)).map(|i| (v, i, false)).collect());
                    let want = Array2::from_shape_fn((rows, stage.len()), |(t, b)| {
                        stage[b].iter().map(|&(site, i, logit)| {
                            let read = x.row(t).dot(&frames[site].0.column(i));
                            let weight = if logit { (0..heads * hd).map(|j| frames[site].1[[i, j]].powi(2) * spread[[t, j]]).sum::<f64>() * c * c } else { 1.0 };
                            read * read * weight
                        }).sum()
                    });
                    (trace.values[sum].clone(), want)
                })
                .collect()
        };
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        let device = Device::host();
        let clean: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 }).collect();
        let score = |m: &OperatorProgram, explanation: &crate::library_mdl::Explanation, artifact: &crate::artifact::Artifact| -> f64 {
            let blocks: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
            Interchange::new(&device, m, &blocks, artifact, &explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments").evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum()
        };
        let close = |what: &str, got: &Array2<f64>, want: &Array2<f64>| {
            let err = (got - want).iter().fold(0.0_f64, |m, e| m.max(e.abs()));
            assert!(err <= 1e-8 * want.iter().fold(1.0_f64, |m, e| m.max(e.abs())), "{what}: the gate inputs differ from their definition by {err}");
        };
        let open = build("logits_open", &[]);
        let bits = score(&native, &open, &open.artifact);
        assert!(bits.abs() < 1e-5, "every gate open: KL(M ‖ P) = {bits} bits");
        let inputs = gate_inputs(&open);
        for (l, (got, want)) in inputs.iter().enumerate() {
            close(&format!("open layer {l}"), got, want);
            assert!(want.column(0).iter().filter(|v| **v > 1e-6).count() > want.nrows() / 2, "layer {l}: a logit read's spread is not zero");
        }
        // Thresholds between each component's two middle distinct gate inputs (`τ|τ|` against the
        // squared read): the tiny vocabulary repeats inputs, and no token may sit on its threshold.
        let taus: Vec<Vec<f64>> = inputs.iter().map(|(got, _)| got.columns().into_iter().map(|col| {
            let mut v = col.to_vec();
            v.sort_by(f64::total_cmp);
            v.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * b.abs().max(1e-12));
            (0.5 * (v[v.len() / 2] + v[(v.len() / 2).saturating_sub(1)])).sqrt()
        }).collect()).collect();
        let gated = build("logits_gated", &taus);
        for (l, (got, want)) in gate_inputs(&gated).iter().enumerate() {
            close(&format!("gated layer {l}"), got, want);
            let shut = got.rows().into_iter().map(|r| r.iter().zip(&taus[l]).filter(|(g, t)| **g <= **t * **t).count()).sum::<usize>();
            assert!(shut > 0 && shut < got.len(), "layer {l}: some components shut ({shut} of {})", got.len());
        }
        let shut_bits = score(&native, &gated, &gated.artifact);
        assert!(shut_bits > 1e-3, "some components shut: KL(M ‖ P) = {shut_bits} bits");
        // The single-precision device (Metal on the Mac) shuts the same gates.
        if let Some(f32_device) = Device::single_precision(gam_gpu::GpuPolicy::Auto).expect("the device query") {
            let blocks: Vec<_> = gated.layers.iter().map(|l| l.sites.clone()).collect();
            let bits: f64 = Interchange::new(&f32_device, &native, &blocks, &gated.artifact, &gated.trainable, Vec::new(), 1 << 30, 64).expect("the experiments").evaluate(&batch, &clean, false).expect("evaluate").bits.iter().flatten().sum();
            assert!((bits - shut_bits).abs() <= 1e-3 * shut_bits.max(1.0), "{}: KL(M ‖ P) = {bits} bits against the host's {shut_bits}", f32_device.name());
        }
        // Head edits: every head's q, k, v rows and o columns scaled, through the training path and
        // by direct mutation of M's maps and P's writes and o reads.
        let edits: Vec<crate::weight_edit::Drawn> = crate::weight_edit::candidates(&native, 3, 60).expect("the edits").into_iter().filter(|d| d.kind == crate::weight_edit::Kind::Head).take(4).collect();
        assert!(edits.len() == 4, "head edits drawn");
        let blocks: Vec<_> = gated.layers.iter().map(|l| l.sites.clone()).collect();
        let mut ic = Interchange::new(&device, &native, &blocks, &gated.artifact, &gated.trainable, Vec::new(), 1 << 30, 64).expect("the experiments");
        ic.set_weight_edits(edits.clone()).expect("the table");
        let scaled = |program: &mut OperatorProgram, name: &str, rows: bool, units: std::ops::Range<usize>, alpha: f64| {
            let op = super::index_of(program, name).expect("the operator");
            let old = std::sync::Arc::clone(&program.operators[op]);
            let mut v = old.matrix();
            for u in units {
                if rows { v.row_mut(u).mapv_inplace(|x| x * alpha) } else { v.column_mut(u).mapv_inplace(|x| x * alpha) }
            }
            let precision = crate::operator_program::exact_precision(v.iter().copied()).expect("a precision");
            program.operators[op] = std::sync::Arc::new(crate::operator_program::Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), v, precision, old.provenance.clone()).expect("an operator"));
        };
        for (i, d) in edits.iter().enumerate() {
            let experiments: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: Some(crate::interchange::Patch::Weights { edit: i, block: d.block }), position: 0 }).collect();
            let training: f64 = ic.evaluate(&batch, &experiments, false).expect("evaluate").bits.iter().flatten().sum();
            let (mut m, mut p) = (native.clone(), gated.artifact.clone());
            for e in &d.entries {
                let width = if e.rows { native.operators[super::index_of(&native, &e.native).expect("a map")].rows.width() } else { native.operators[super::index_of(&native, &e.native).expect("a map")].cols.width() };
                scaled(&mut m, &e.native, e.rows, 0..width, e.alpha);
                let mut parts = e.native.split('.');
                let (l, map) = (parts.nth(1).expect("a layer"), parts.next().expect("a map"));
                let (kind, h): (&str, usize) = (&map[..1], map[1..].parse().expect("a head"));
                if kind == "o" {
                    scaled(&mut p.program, &format!("library.l{l}.o.read"), false, h * width..(h + 1) * width, e.alpha);
                } else {
                    scaled(&mut p.program, &format!("library.l{l}.h{h}.{kind}"), true, 0..width, e.alpha);
                }
            }
            let direct = score(&m, &gated, &p);
            assert!((training - direct).abs() <= 1e-9 * direct.abs().max(1.0), "edit {i}: training {training} bits, direct mutation {direct}");
        }
        std::fs::remove_dir_all(&factors).expect("the factors are removed");
    }
}
