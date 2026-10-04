//! Known-mechanism toys (#2951): small models whose mechanism is known, each with counterfactual
//! questions, to score an explanation the way CHIVE does (an explanation is good when it predicts
//! what the model does on counterfactual inputs and under interventions without running the model)
//! and to check that its decomposition contains the known mechanism.
//!
//! A case (written by `bench/toys_2951`) is an engine export, `questions.json`, `truth.json` and
//! `transcript.f64`. A `transformer` export is imported position by position
//! ([`super::import::import_rows`], the layout of a language model: one row per position, one site
//! per map shared by every position); a `residual_mlp` export as one row per input. A question is a
//! prompt and its edits: a new prompt, or a native intervention, `scale` (the value at a place
//! times a factor) or `patch` (the value at a place set to its value on a donor prompt in the same
//! program). Places are native ([`Place`]): a head's attention read (before its output map), an
//! MLP's ReLU outputs, or the residual after a block (before block 0 at layer −1), at one position
//! or all. The measured outcome is the native model's logits under the edits at the readout
//! positions; the held-out split's outcomes live in `<root>/sealed/<case>.json`, which only
//! [`sealed`] reads.
//!
//! # Predictions
//!
//! An explanation predicts a question by running its own program under the same edits at the same
//! places ([`Explained`]): its sites replaced by their libraries, each site's blocks chosen by its
//! selection from the reads the explanation itself computed, the edits applied in node order, a
//! donor's value taken from the explanation's own run on the donor. Any other explainer is scored
//! from a file of logits per question ([`score`]). The baselines know no internals: `null`
//! predicts the clean outcome (no edit matters), `transcript` the clean outcome when the prompt is
//! unchanged and otherwise the mean distribution of the transcript prompts nearest the edited one.
//!
//! # Mechanism
//!
//! The truth names weight components (parts of site tensors summing to them), the probe prompts on
//! which a component is active, and attention targets. Against an explanation's library:
//!
//! * purity: each block `M_b = U_bᵀ V_b` of a site is assigned the component whose subspace
//!   (`P_col M P_row`, the column and row spaces of the component's part of the site) holds the
//!   largest share of `‖M_b‖²_F`; a site's purity is that share averaged with weights `‖M_b‖²_F`;
//! * coverage: the component's part of the site carried by the blocks assigned to it,
//!   `⟨Σ_b M_b, T_c⟩ / ‖T_c‖²_F`;
//! * activity: the F1 of the probe prompts on which a component's blocks (assigned at a site) run
//!   at a readout position against its active prompts; a component with no weights is matched to
//!   the single best block of any site;
//! * attention: the explanation's own attention weight on each target key, averaged.
//!
//! A mechanism is recovered when each holds by a majority: purity and coverage above 1/2, activity
//! F1 above 1/2, attention above 1/2 on the targets.

use super::dense::svd;
use super::explanation::{Explanation, Replacement};
use super::import::{Imported, import, import_rows, sequence_rows};
use super::masked::{Masked, Site};
use super::operator_program::{FamilyInputs, Node, OperatorProgram, SlotValues, Trace};
use ndarray::{Array1, Array2, Axis, s};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What a place names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// A head's attention read (before `W_O`), `d_head` wide.
    Head,
    /// An MLP's ReLU outputs, one per unit.
    Mlp,
    /// The residual after block `layer` (before block 0 at −1).
    Resid,
}

/// A native place: every position (and unit) when none is named.
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub kind: Kind,
    pub layer: i64,
    pub head: Option<usize>,
    pub position: Option<usize>,
    pub units: Option<Vec<usize>>,
}

/// One edit of a question.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    Input(Vec<f64>),
    Scale { place: Place, factor: f64 },
    Patch { place: Place, donor: Vec<f64> },
    /// The span of `directions` (rows, the place's width) removed from the value at `place`.
    Project { place: Place, directions: Array2<f64> },
}

/// A counterfactual question: logits are readouts × classes.
#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub tag: String,
    pub held_out: bool,
    pub input: Vec<f64>,
    pub edits: Vec<Edit>,
    pub clean: Array2<f64>,
    /// The measured outcome (the dev split only).
    pub outcome: Option<Array2<f64>>,
}

impl Question {
    /// The prompt the model reads: the last input edit's, else the question's.
    pub fn prompt(&self) -> &[f64] {
        self.edits.iter().rev().find_map(|e| if let Edit::Input(x) = e { Some(x.as_slice()) } else { None }).unwrap_or(&self.input)
    }

    /// Whether any edit acts inside the model.
    pub fn internal(&self) -> bool {
        self.edits.iter().any(|e| !matches!(e, Edit::Input(..)))
    }
}

/// A weight component of the known mechanism.
#[derive(Clone, Debug)]
pub struct Component {
    pub name: String,
    /// Per program operator name, its part (in the operator's orientation).
    pub pieces: BTreeMap<String, Array2<f64>>,
    /// The probe prompts on which it is active.
    pub active: Option<Vec<usize>>,
}

/// An attention target set: per probe prompt, `(query position, key position)` pairs.
#[derive(Clone, Debug)]
pub struct Attention {
    pub name: String,
    pub layer: i64,
    pub head: usize,
    pub targets: Vec<Vec<(usize, usize)>>,
}

/// The known mechanism.
#[derive(Clone, Debug)]
pub struct Truth {
    pub mechanism: String,
    pub components: Vec<Component>,
    pub attention: Vec<Attention>,
    pub probe: Vec<Vec<f64>>,
}

/// A head's query and key nodes and whether it is causal.
#[derive(Clone, Copy, Debug)]
struct Head {
    query: usize,
    key: usize,
    causal: bool,
}

/// Where each native place lives in the imported program.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    /// `(kind, layer, head)` → node (head 0 for MLPs and residuals).
    nodes: BTreeMap<(Kind, i64, usize), usize>,
    heads: BTreeMap<(i64, usize), Head>,
}

/// `blocks.{l}.{stem}{h}` → `(l, stem, h)`; `W_E` → `(−1, "W_E", None)`.
fn operator_place(name: &str) -> Option<(i64, String, Option<usize>)> {
    if name == "W_E" {
        return Some((-1, name.to_string(), None));
    }
    let rest = name.strip_prefix("blocks.")?;
    let (layer, stem) = rest.split_once('.')?;
    let digits = stem.len() - stem.chars().rev().take_while(char::is_ascii_digit).count();
    Some((layer.parse().ok()?, stem[..digits].to_string(), stem[digits..].parse::<usize>().ok()))
}

impl Layout {
    /// The places of a program from [`import_rows`] or a `residual_mlp` [`import`], read off its
    /// operators' names.
    pub fn of(program: &OperatorProgram) -> Self {
        let named = |op: usize| operator_place(&program.operators[op].name);
        let first = |node: usize| match &program.nodes[node] {
            Node::Affine { terms, .. } => terms.first().and_then(|(_, op)| named(*op)),
            _ => None,
        };
        let mut layout = Self::default();
        let (mut after_attention, mut after_mlp) = (BTreeMap::new(), BTreeMap::new());
        for (index, node) in program.nodes.iter().enumerate() {
            match node {
                Node::Affine { terms, .. } => {
                    for (_, op) in terms {
                        match named(*op) {
                            Some((-1, stem, None)) if stem == "W_E" => {
                                layout.nodes.insert((Kind::Resid, -1, 0), index);
                            }
                            Some((l, stem, None)) if stem == "W_out" => {
                                after_mlp.insert(l, index);
                            }
                            Some((l, stem, Some(_))) if stem == "W_O" => {
                                after_attention.insert(l, index);
                            }
                            _ => continue,
                        }
                    }
                }
                Node::Pointwise { input, .. } => {
                    if let Some((l, stem, None)) = first(*input)
                        && stem == "W_in"
                    {
                        layout.nodes.insert((Kind::Mlp, l, 0), index);
                    }
                }
                Node::Attend { query, key, causal, .. } => {
                    if let Some((l, stem, Some(h))) = first(*query)
                        && stem == "W_Q"
                    {
                        layout.nodes.insert((Kind::Head, l, h), index);
                        layout.heads.insert((l, h), Head { query: *query, key: *key, causal: *causal });
                    }
                }
                _ => continue,
            }
        }
        for (l, node) in after_attention {
            layout.nodes.entry((Kind::Resid, l, 0)).or_insert(node);
        }
        for (l, node) in after_mlp {
            layout.nodes.insert((Kind::Resid, l, 0), node);
        }
        layout
    }

    /// The node `place` names.
    pub fn node(&self, place: &Place) -> Result<usize, String> {
        let head = if place.kind == Kind::Head { place.head.ok_or("a head place names no head")? } else { 0 };
        self.nodes.get(&(place.kind, place.layer, head)).copied().ok_or_else(|| format!("{place:?}: no such place in the program"))
    }
}

/// An edit resolved to a program node: its rows and coordinates (all when none) scaled, set to a
/// donor's values (rows × width), or with an orthonormal set of directions (rows) projected out.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub node: usize,
    pub rows: Option<Vec<usize>>,
    pub coords: Option<Vec<usize>>,
    pub action: Action,
}

#[derive(Clone, Debug)]
pub enum Action {
    Scale(f64),
    Set(Array2<f64>),
    Remove(Array2<f64>),
}

/// An orthonormal basis (rows) of the span of `directions`' rows (modified Gram–Schmidt; a row in
/// the span of the earlier ones adds nothing).
fn orthonormal(directions: &Array2<f64>) -> Array2<f64> {
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let largest = directions.outer_iter().map(|r| r.dot(&r).sqrt()).fold(0.0_f64, f64::max);
    for row in directions.outer_iter() {
        let mut v = row.to_owned();
        for q in &basis {
            let along = v.dot(q);
            v.scaled_add(-along, q);
        }
        let norm = v.dot(&v).sqrt();
        if norm > f64::EPSILON * largest * directions.ncols() as f64 {
            basis.push(v / norm);
        }
    }
    let width = directions.ncols();
    Array2::from_shape_fn((basis.len(), width), |(i, j)| basis[i][j])
}

fn apply(value: &mut Array2<f64>, edit: &Resolved) -> Result<(), String> {
    let rows: Vec<usize> = edit.rows.clone().unwrap_or_else(|| (0..value.nrows()).collect());
    let columns: Vec<usize> = edit.coords.clone().unwrap_or_else(|| (0..value.ncols()).collect());
    if rows.iter().any(|r| *r >= value.nrows()) || columns.iter().any(|c| *c >= value.ncols()) {
        return Err(format!("rows {rows:?}, coordinates {columns:?} of a {:?} node", value.dim()));
    }
    if let Action::Set(donor) = &edit.action
        && donor.dim() != value.dim()
    {
        return Err(format!("a donor of {:?} for a node of {:?}", donor.dim(), value.dim()));
    }
    if let Action::Remove(basis) = &edit.action {
        if basis.ncols() != value.ncols() || edit.coords.is_some() {
            return Err(format!("{} directions of width {} for a {}-wide node", basis.nrows(), basis.ncols(), value.ncols()));
        }
        for &r in &rows {
            for q in basis.outer_iter() {
                let along = value.row(r).dot(&q);
                value.row_mut(r).scaled_add(-along, &q);
            }
        }
        return Ok(());
    }
    for &r in &rows {
        for &c in &columns {
            value[[r, c]] = match &edit.action {
                Action::Scale(factor) => value[[r, c]] * factor,
                Action::Set(donor) => donor[[r, c]],
                Action::Remove(..) => value[[r, c]],
            };
        }
    }
    Ok(())
}

/// One forward of `program` on `family` with `edits` applied in node order, and `gate(node,
/// trace)` called at each of `gates` (program nodes) once every node before it is final; every
/// node after an edit or gate is recomputed.
fn sweep(
    program: &OperatorProgram,
    family: &FamilyInputs,
    edits: &[Resolved],
    gates: &[usize],
    mut gate: impl FnMut(usize, &mut Trace) -> Result<(), String>,
) -> Result<Trace, String> {
    let mut events: BTreeMap<usize, (Vec<&Resolved>, bool)> = BTreeMap::new();
    for e in edits {
        events.entry(e.node).or_default().0.push(e);
    }
    for g in gates {
        events.entry(*g).or_default().1 = true;
    }
    let mut trace = program.execute(family, false).map_err(|e| e.to_string())?;
    for (node, (at, gated)) in events {
        for e in at {
            apply(&mut trace.values[node], e)?;
        }
        if gated {
            gate(node, &mut trace)?;
        }
        program.execute_from(family, &mut trace, node + 1).map_err(|e| e.to_string())?;
    }
    Ok(trace)
}

/// A program a question runs in: the native model, or an explanation's.
pub trait Runner {
    /// The program node of the imported program's node `node`.
    fn node(&self, node: usize) -> usize;
    /// One forward on `family` (the imported program's slots) under `edits`.
    fn run(&self, family: &FamilyInputs, edits: &[Resolved]) -> Result<Trace, String>;
    /// The node holding the logits.
    fn output(&self) -> usize;
}

/// The native model.
pub struct Native<'a>(pub &'a OperatorProgram);

impl Runner for Native<'_> {
    fn node(&self, node: usize) -> usize {
        node
    }

    fn run(&self, family: &FamilyInputs, edits: &[Resolved]) -> Result<Trace, String> {
        sweep(self.0, family, edits, &[], |_, _| Ok(()))
    }

    fn output(&self) -> usize {
        self.0.output
    }
}

/// An explanation run autonomously: every site replaced, each site's blocks chosen from the reads
/// its own program computed under the edits.
pub struct Explained<'a> {
    pub explanation: &'a Explanation,
    pub masked: Masked,
    /// The imported program's node → the masked program's: every site's mask, `z` and `z ⊙ m`
    /// sit just before its first written node.
    map: Vec<usize>,
}

impl<'a> Explained<'a> {
    pub fn new(model: &OperatorProgram, explanation: &'a Explanation) -> Result<Self, String> {
        let members: Vec<usize> = (0..explanation.sites.len()).collect();
        let masked = explanation.masked(model, &members)?;
        let firsts: Vec<usize> = explanation.sites.iter().map(|f| f.site.writes.iter().copied().min().unwrap_or(usize::MAX)).collect();
        let map = (0..model.nodes.len()).map(|n| n + 3 * firsts.iter().filter(|f| **f <= n).count()).collect();
        Ok(Self { explanation, masked, map })
    }

    /// The trace and every site's blocks on (rows × blocks), on `family` under `edits`.
    pub fn run_with_sets(&self, family: &FamilyInputs, edits: &[Resolved]) -> Result<(Trace, Vec<Array2<f64>>), String> {
        let gates = self.masked.gates();
        if gates.len() != self.explanation.sites.len() {
            return Err("a masked site without its gate".to_string());
        }
        let mut chosen: Vec<Array2<f64>> = (0..gates.len()).map(|k| Array2::zeros((family.rows, self.masked.blocks(k)))).collect();
        let full = self.masked.family(family, &chosen);
        let amplitudes: Vec<usize> = gates.iter().map(|(z, _)| *z).collect();
        let trace = sweep(&self.masked.program, &full, edits, &amplitudes, |node, trace| {
            let k = amplitudes.iter().position(|z| *z == node).ok_or("an unknown gate")?;
            let views: Vec<_> = self.masked.sites[k].reads.iter().map(|n| trace.values[*n].view()).collect();
            let reads = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
            let on = self.explanation.sites[k].select(&reads);
            trace.values[gates[k].1] = self.masked.expand(k, &on);
            chosen[k] = on;
            Ok(())
        })?;
        Ok((trace, chosen))
    }
}

impl Runner for Explained<'_> {
    fn node(&self, node: usize) -> usize {
        self.map[node]
    }

    fn run(&self, family: &FamilyInputs, edits: &[Resolved]) -> Result<Trace, String> {
        self.run_with_sets(family, edits).map(|(trace, _)| trace)
    }

    fn output(&self) -> usize {
        self.masked.program.output
    }
}

/// A case: its imported model, places, questions, truth and transcript.
pub struct Case {
    pub name: String,
    pub dir: PathBuf,
    pub imported: Imported,
    pub layout: Layout,
    pub questions: Vec<Question>,
    pub truth: Truth,
    /// Transcript prompts (rows × prompt width).
    pub transcript: Array2<f64>,
    /// The readout positions (a `transformer`'s; the one row of a residual MLP's).
    pub readouts: Vec<usize>,
    /// Readouts and classes of the logits.
    pub shape: (usize, usize),
}

fn floats(value: &Value) -> Result<Vec<f64>, String> {
    value.as_array().ok_or("not a list")?.iter().map(|v| v.as_f64().ok_or_else(|| format!("{v} is not a number"))).collect()
}

fn matrix_of(value: &Value) -> Result<Array2<f64>, String> {
    let rows: Vec<Vec<f64>> = value.as_array().ok_or("logits are not a list")?.iter().map(floats).collect::<Result<_, _>>()?;
    let cols = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|r| r.len() != cols) {
        return Err("ragged logits".to_string());
    }
    Array2::from_shape_vec((rows.len(), cols), rows.concat()).map_err(|e| e.to_string())
}

fn indices(value: &Value) -> Result<Vec<usize>, String> {
    value.as_array().ok_or("not a list")?.iter().map(|v| v.as_u64().map(|u| u as usize).ok_or_else(|| format!("{v} is not an index"))).collect()
}

fn place_of(value: &Value) -> Result<Place, String> {
    let kind = match value["node"].as_str() {
        Some("head") => Kind::Head,
        Some("mlp") => Kind::Mlp,
        Some("resid") => Kind::Resid,
        other => return Err(format!("unknown place {other:?}")),
    };
    let index = |key: &str| value.get(key).and_then(Value::as_u64).map(|v| v as usize);
    let units = value.get("units").map(indices).transpose()?;
    Ok(Place { kind, layer: value["layer"].as_i64().ok_or("a place without a layer")?, head: index("head"), position: index("position"), units })
}

fn edit_of(value: &Value) -> Result<Edit, String> {
    match value["kind"].as_str() {
        Some("input") => Ok(Edit::Input(floats(&value["input"])?)),
        Some("scale") => Ok(Edit::Scale { place: place_of(&value["place"])?, factor: value["factor"].as_f64().ok_or("a scale without a factor")? }),
        Some("patch") => Ok(Edit::Patch { place: place_of(&value["place"])?, donor: floats(&value["donor"])? }),
        Some("project") => Ok(Edit::Project { place: place_of(&value["place"])?, directions: matrix_of(&value["directions"])? }),
        other => Err(format!("unknown edit {other:?}")),
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| format!("{}: {e}", path.display()))
}

/// A tensor's part as the program's operators see it: `W_Q/K/V` split by head rows, `W_O` by head
/// columns, a residual MLP's `W_E` transposed, anything else whole.
fn operator_pieces(tensor: &str, piece: &Array2<f64>, kind: &str, d_head: usize) -> Vec<(String, Array2<f64>)> {
    let per_head = |by_rows: bool| -> Vec<(String, Array2<f64>)> {
        let width = if by_rows { piece.nrows() } else { piece.ncols() };
        (0..width / d_head.max(1))
            .map(|h| {
                let part = if by_rows { piece.slice(s![h * d_head..(h + 1) * d_head, ..]) } else { piece.slice(s![.., h * d_head..(h + 1) * d_head]) };
                (format!("{tensor}{h}"), part.to_owned())
            })
            .collect()
    };
    match operator_place(tensor) {
        Some((_, stem, None)) if ["W_Q", "W_K", "W_V"].contains(&stem.as_str()) => per_head(true),
        Some((_, stem, None)) if stem == "W_O" => per_head(false),
        Some((-1, _, None)) if kind == "residual_mlp" => vec![(tensor.to_string(), piece.t().to_owned())],
        _ => vec![(tensor.to_string(), piece.clone())],
    }
}

fn truth_of(record: &Value, dir: &Path, kind: &str, shapes: &Value, d_head: usize) -> Result<Truth, String> {
    let components = record["components"]
        .as_array()
        .ok_or("truth.json: components")?
        .iter()
        .map(|c| {
            let mut pieces = BTreeMap::new();
            if let Some(named) = c["pieces"].as_object() {
                for (tensor, file) in named {
                    let cols = shapes[tensor]["shape"][1].as_u64().ok_or_else(|| format!("{tensor}: no shape"))? as usize;
                    let piece = super::counterfactual::read_f64_matrix(&dir.join(format!("{}.f64", file.as_str().ok_or("a piece file")?)), cols)?;
                    pieces.extend(operator_pieces(tensor, &piece, kind, d_head));
                }
            }
            let active = c.get("active").map(indices).transpose()?;
            Ok(Component { name: c["name"].as_str().unwrap_or("").to_string(), pieces, active })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let probe: Vec<Vec<f64>> = match record.get("probe") {
        Some(rows) => rows.as_array().ok_or("truth.json: probe")?.iter().map(floats).collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    let pairs = |v: &Value| -> Result<Vec<(usize, usize)>, String> {
        v.as_array().ok_or("attention targets")?.iter().map(|p| Ok((p[0].as_u64().ok_or("a query")? as usize, p[1].as_u64().ok_or("a key")? as usize))).collect()
    };
    let attention = record["attention"]
        .as_array()
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .map(|a| {
            let targets = match a.get("per_probe") {
                Some(per) => per.as_array().ok_or("per_probe")?.iter().map(pairs).collect::<Result<_, _>>()?,
                None => vec![pairs(&a["targets"])?; probe.len()],
            };
            Ok(Attention {
                name: a["name"].as_str().unwrap_or("").to_string(),
                layer: a["layer"].as_i64().ok_or("attention layer")?,
                head: a["head"].as_u64().ok_or("attention head")? as usize,
                targets,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Truth { mechanism: record["mechanism"].as_str().unwrap_or("").to_string(), components, attention, probe })
}

impl Case {
    /// The case in `dir` (its public files only).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let export = read_json(&dir.join("export.json"))?;
        let imported = if export["kind"].as_str() == Some("transformer") { import_rows(dir)? } else { import(dir)? };
        let layout = Layout::of(&imported.program);
        let record = &imported.record;
        let kind = imported.kind.clone();
        let (readouts, shape) = match kind.as_str() {
            "transformer_rows" => {
                let readouts = indices(&record["output"]["readout_positions"])?;
                let classes = record["output"]["n_classes"].as_u64().ok_or("n_classes")? as usize;
                let shape = (readouts.len(), classes);
                (readouts, shape)
            }
            "residual_mlp" => (vec![0], (record["output"]["n_outputs"].as_u64().ok_or("n_outputs")? as usize, 2)),
            other => return Err(format!("{other}: not a toy kind")),
        };
        let listed = read_json(&dir.join("questions.json"))?;
        let questions = listed["questions"]
            .as_array()
            .ok_or("questions.json: questions")?
            .iter()
            .map(|q| {
                Ok(Question {
                    id: q["id"].as_str().ok_or("a question without an id")?.to_string(),
                    tag: q["tag"].as_str().unwrap_or("").to_string(),
                    held_out: q["split"].as_str() == Some("held_out"),
                    input: floats(&q["input"])?,
                    edits: q["edits"].as_array().ok_or("a question without edits")?.iter().map(edit_of).collect::<Result<_, _>>()?,
                    clean: matrix_of(&q["clean"])?,
                    outcome: q.get("outcome").map(matrix_of).transpose()?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let width = record["samples"]["shape"][1].as_u64().ok_or("samples.shape")? as usize;
        let transcript = super::counterfactual::read_f64_matrix(&dir.join("transcript.f64"), width)?;
        let d_head = record["config"]["d_head"].as_u64().unwrap_or(0) as usize;
        let truth = truth_of(&read_json(&dir.join("truth.json"))?, dir, &kind, &record["files"], d_head)?;
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("case").to_string();
        Ok(Self { name, dir: dir.to_path_buf(), imported, layout, questions, truth, transcript, readouts, shape })
    }

    /// Whether the program runs position by position (a transformer).
    fn per_position(&self) -> bool {
        self.imported.kind == "transformer_rows"
    }

    /// The family of `prompts` in the program's rows and slots.
    pub fn family(&self, prompts: &[&[f64]]) -> Result<FamilyInputs, String> {
        let width = prompts.first().map_or(0, |p| p.len());
        if prompts.iter().any(|p| p.len() != width) {
            return Err("prompts of different lengths".to_string());
        }
        let matrix = Array2::from_shape_fn((prompts.len(), width), |(r, c)| prompts[r][c]);
        if self.per_position() {
            return Ok(sequence_rows(&matrix));
        }
        Ok(FamilyInputs { rows: prompts.len(), slots: vec![SlotValues::Raw(matrix)], layout: None })
    }

    /// `question`'s edits in `runner`'s nodes (a donor's values from `runner`'s own run on it).
    pub fn resolve(&self, runner: &dyn Runner, question: &Question) -> Result<Vec<Resolved>, String> {
        let mut resolved = Vec::new();
        for edit in &question.edits {
            let (place, action) = match edit {
                Edit::Input(..) => continue,
                Edit::Scale { place, factor } => (place, Action::Scale(*factor)),
                Edit::Patch { place, donor } => {
                    let node = runner.node(self.layout.node(place)?);
                    (place, Action::Set(runner.run(&self.family(&[donor.as_slice()])?, &[])?.values[node].clone()))
                }
                Edit::Project { place, directions } => (place, Action::Remove(orthonormal(directions))),
            };
            if place.position.is_some() && !self.per_position() {
                return Err(format!("{place:?}: a position in a model without positions"));
            }
            // A projection acts on the whole value; a unit list selects coordinates otherwise.
            let coords = if place.kind == Kind::Mlp && !matches!(action, Action::Remove(..)) { place.units.clone() } else { None };
            resolved.push(Resolved { node: runner.node(self.layout.node(place)?), rows: place.position.map(|p| vec![p]), coords, action });
        }
        Ok(resolved)
    }

    /// The logits at the readouts (readouts × classes) of prompt `prompt` in an output `logits` of
    /// prompts `length` long.
    fn readout(&self, logits: &Array2<f64>, prompt: usize, length: usize) -> Result<Array2<f64>, String> {
        if self.per_position() {
            let rows: Vec<usize> = self.readouts.iter().map(|r| prompt * length + r).collect();
            return Ok(logits.select(Axis(0), &rows));
        }
        Array2::from_shape_vec(self.shape, logits.row(prompt).to_vec()).map_err(|e| e.to_string())
    }

    /// `runner`'s logits (readouts × classes) for `question`.
    pub fn predict(&self, runner: &dyn Runner, question: &Question) -> Result<Array2<f64>, String> {
        let edits = self.resolve(runner, question)?;
        let trace = runner.run(&self.family(&[question.prompt()])?, &edits)?;
        self.readout(&trace.values[runner.output()], 0, question.prompt().len())
    }

    /// The native logits (readouts × classes) of every prompt of `prompts`.
    pub fn native_logits(&self, prompts: &Array2<f64>) -> Result<Vec<Array2<f64>>, String> {
        if prompts.nrows() == 0 {
            return Ok(Vec::new());
        }
        let rows: Vec<Vec<f64>> = prompts.outer_iter().map(|r| r.to_vec()).collect();
        let views: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let model = &self.imported.program;
        let logits = model.execute(&self.family(&views)?, false).map_err(|e| e.to_string())?.values.swap_remove(model.output);
        (0..prompts.nrows()).map(|p| self.readout(&logits, p, prompts.ncols())).collect()
    }

    /// The transcript baseline: the clean logits when the prompt is unchanged, else the log of the
    /// mean distribution over the transcript prompts nearest the edited one (Hamming distance on
    /// tokens, Euclidean on reals; every tie), `outputs` the native logits of each transcript row.
    pub fn transcript_prediction(&self, outputs: &[Array2<f64>], question: &Question) -> Array2<f64> {
        let prompt = question.prompt();
        if prompt == question.input.as_slice() || outputs.is_empty() {
            return question.clean.clone();
        }
        let tokens = self.per_position();
        let distance = |row: ndarray::ArrayView1<f64>| -> f64 {
            row.iter().zip(prompt).map(|(a, b)| if tokens { f64::from(u8::from(a != b)) } else { (a - b) * (a - b) }).sum()
        };
        let distances: Vec<f64> = self.transcript.outer_iter().map(distance).collect();
        let nearest = distances.iter().copied().fold(f64::INFINITY, f64::min);
        let mut mean = Array2::<f64>::zeros(question.clean.dim());
        let mut count = 0.0;
        for (logits, d) in outputs.iter().zip(&distances) {
            if *d == nearest {
                mean += &softmax_rows(logits);
                count += 1.0;
            }
        }
        (mean / count).mapv(f64::ln)
    }

    /// Per probe prompt, the attention weights (positions × positions) of head `head` of layer
    /// `layer` in `trace` (a run of `runner` on the probe), from its query and key reads.
    fn patterns(&self, trace: &Trace, runner: &dyn Runner, layer: i64, head: usize) -> Result<Vec<Array2<f64>>, String> {
        let h = self.layout.heads.get(&(layer, head)).ok_or_else(|| format!("no head {layer}.{head}"))?;
        let (q, k) = (&trace.values[runner.node(h.query)], &trace.values[runner.node(h.key)]);
        let length = self.truth.probe.first().map_or(0, Vec::len);
        let scale = 1.0 / (q.ncols() as f64).sqrt();
        Ok((0..self.truth.probe.len())
            .map(|p| {
                let rows = s![p * length..(p + 1) * length, ..];
                let scores = q.slice(rows).dot(&k.slice(rows).t()) * scale;
                let masked = Array2::from_shape_fn((length, length), |(i, j)| if h.causal && j > i { f64::NEG_INFINITY } else { scores[[i, j]] });
                softmax_rows(&masked)
            })
            .collect())
    }
}

fn softmax_rows(logits: &Array2<f64>) -> Array2<f64> {
    let mut p = logits.clone();
    for mut row in p.outer_iter_mut() {
        let top = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - top).exp());
        let total = row.sum();
        row.mapv_inplace(|v| v / total);
    }
    p
}

/// `KL(p ‖ q)` per readout between the distributions of logits `p` and `q`.
pub fn kl_rows(p: &Array2<f64>, q: &Array2<f64>) -> Array1<f64> {
    let (sp, sq) = (softmax_rows(p), softmax_rows(q));
    Array1::from_iter(sp.outer_iter().zip(sq.outer_iter()).map(|(a, b)| a.iter().zip(b).map(|(x, y)| if *x > 0.0 { x * (x.ln() - y.max(f64::MIN_POSITIVE).ln()) } else { 0.0 }).sum()))
}

fn argmax(row: ndarray::ArrayView1<f64>) -> usize {
    row.iter().enumerate().fold((0, f64::NEG_INFINITY), |best, (i, v)| if *v > best.1 { (i, *v) } else { best }).0
}

/// An explainer's agreement with the measured outcomes on a set of questions.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Scores {
    pub questions: usize,
    /// Mean over questions of the mean over readouts of `KL(native ‖ prediction)`.
    pub mean_kl: f64,
    /// The fraction of readouts whose predicted argmax is the native one.
    pub argmax: f64,
    /// Readouts whose native argmax the edit moved from the clean one, and the fraction of those
    /// whose argmax the prediction gets.
    pub changed: usize,
    pub changed_argmax: f64,
}

/// `predictions` against `outcomes` (by question id) over `questions`; a question with no
/// prediction counts as the uniform distribution.
pub fn score(questions: &[&Question], outcomes: &BTreeMap<String, Array2<f64>>, predictions: &BTreeMap<String, Array2<f64>>) -> Result<Scores, String> {
    let mut s = Scores::default();
    let (mut kl, mut agree, mut readouts, mut changed_agree) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for q in questions {
        let outcome = outcomes.get(&q.id).ok_or_else(|| format!("{}: no outcome", q.id))?;
        let uniform = Array2::zeros(outcome.dim());
        let predicted = predictions.get(&q.id).unwrap_or(&uniform);
        if predicted.dim() != outcome.dim() {
            return Err(format!("{}: predicted {:?}, measured {:?}", q.id, predicted.dim(), outcome.dim()));
        }
        kl += kl_rows(outcome, predicted).mean().unwrap_or(0.0);
        for ((o, p), c) in outcome.outer_iter().zip(predicted.outer_iter()).zip(q.clean.outer_iter()) {
            let hit = f64::from(u8::from(argmax(o) == argmax(p)));
            agree += hit;
            readouts += 1.0;
            if argmax(o) != argmax(c) {
                s.changed += 1;
                changed_agree += hit;
            }
        }
        s.questions += 1;
    }
    s.mean_kl = kl / (s.questions.max(1) as f64);
    s.argmax = agree / readouts.max(1.0);
    s.changed_argmax = if s.changed > 0 { changed_agree / s.changed as f64 } else { f64::NAN };
    Ok(s)
}

/// The claims of a CHIVE evaluation (Karvonen et al., arXiv 2608.16747) on a set of questions, with
/// its template and thresholds: per question and readout the behaviour is the clean argmax token,
/// its rate the probability the model gives it, and the claim "this edit changes the behavior rate
/// by ≥ 30 pp" is true when the edit moved the rate by at least 50 pp and false when by at most
/// 15 pp (an edit in between makes no claim). A predictor's confidence in a claim is the change it
/// predicts, `|q(token) − p_clean(token)|`; its score is the AUROC over the claims.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Claims {
    pub true_claims: usize,
    pub false_claims: usize,
    pub auroc: f64,
}

fn rate(logits: ndarray::ArrayView1<f64>, token: usize) -> f64 {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = logits.iter().map(|v| (v - top).exp()).sum();
    (logits[token] - top).exp() / total
}

/// `predictions`' CHIVE claims against `outcomes` over `questions` (a missing prediction predicts
/// no change).
pub fn chive(questions: &[&Question], outcomes: &BTreeMap<String, Array2<f64>>, predictions: &BTreeMap<String, Array2<f64>>) -> Result<Claims, String> {
    let (mut truths, mut falses) = (Vec::new(), Vec::new());
    for q in questions {
        let outcome = outcomes.get(&q.id).ok_or_else(|| format!("{}: no outcome", q.id))?;
        let predicted = predictions.get(&q.id).unwrap_or(&q.clean);
        for r in 0..q.clean.nrows() {
            let token = argmax(q.clean.row(r));
            let before = rate(q.clean.row(r), token);
            let moved = (rate(outcome.row(r), token) - before).abs();
            let confidence = (rate(predicted.row(r), token) - before).abs();
            if moved >= 0.5 {
                truths.push(confidence);
            } else if moved <= 0.15 {
                falses.push(confidence);
            }
        }
    }
    Ok(Claims { true_claims: truths.len(), false_claims: falses.len(), auroc: auroc(&truths, &falses) })
}

/// The AUROC of confidences on true claims against false ones (ties count half).
pub fn auroc(truths: &[f64], falses: &[f64]) -> f64 {
    let pairs = (truths.len() * falses.len()) as f64;
    let wins: f64 = truths.iter().map(|t| falses.iter().map(|f| if t > f { 1.0 } else if t == f { 0.5 } else { 0.0 }).sum::<f64>()).sum();
    if pairs > 0.0 { wins / pairs } else { f64::NAN }
}

/// A CHIVE claim file's predictions (claim id → P(true)) scored by AUROC on its dev split (labels
/// in `<root>/<case>/claims.json`) and its held-out split (labels in `<root>/sealed/<case>.json`,
/// the scorer's only reader), each also per stratum (a claim's public `stratum`, when it has one);
/// a claim with no prediction counts as 1/2.
pub fn chive_claims(root: &Path, case: &str, predictions: &BTreeMap<String, f64>) -> Result<BTreeMap<String, Claims>, String> {
    let record = read_json(&root.join(case).join("claims.json"))?;
    let sealed = read_json(&root.join("sealed").join(format!("{case}.json")))?;
    let mut groups: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for claim in record["claims"].as_array().ok_or("claims.json: claims")? {
        let id = claim["id"].as_str().ok_or("a claim without an id")?;
        let held = claim["split"].as_str() == Some("held_out");
        let label = if held { sealed["labels"][id]["label"].as_bool() } else { claim["label"].as_bool() }.ok_or_else(|| format!("{id}: no label"))?;
        let split = if held { "held_out" } else { "dev" };
        let p = predictions.get(id).copied().unwrap_or(0.5);
        let mut keys = vec![split.to_string()];
        if let Some(stratum) = claim["stratum"].as_str() {
            keys.push(format!("{split}/{stratum}"));
        }
        for key in keys {
            let group = groups.entry(key).or_default();
            if label { group.0.push(p) } else { group.1.push(p) }
        }
    }
    Ok(groups.into_iter().map(|(k, (t, f))| (k, Claims { true_claims: t.len(), false_claims: f.len(), auroc: auroc(&t, &f) })).collect())
}

/// The measured outcomes: the dev split's from the questions, the held-out split's from
/// `<root>/sealed/<case>.json` (the scorer's only reader of the sealed file).
pub fn sealed(root: &Path, case: &Case) -> Result<BTreeMap<String, Array2<f64>>, String> {
    let mut outcomes: BTreeMap<String, Array2<f64>> = case.questions.iter().filter_map(|q| q.outcome.clone().map(|o| (q.id.clone(), o))).collect();
    let record = read_json(&root.join("sealed").join(format!("{}.json", case.name)))?;
    for (id, logits) in record["outcomes"].as_object().ok_or("sealed outcomes")? {
        outcomes.insert(id.clone(), matrix_of(logits)?);
    }
    Ok(outcomes)
}

/// The truth's part of `site` per component touching it (`T_c` in the site's matrix layout).
fn site_parts(program: &OperatorProgram, site: &Site, components: &[Component]) -> Result<Vec<(usize, Array2<f64>)>, String> {
    let interfaces = program.interfaces().map_err(|e| e.to_string())?;
    let offsets = |nodes: &[usize]| -> Vec<usize> {
        nodes
            .iter()
            .scan(0, |at, n| {
                *at += interfaces[*n].width();
                Some(*at - interfaces[*n].width())
            })
            .collect()
    };
    let (ro, wo) = (offsets(&site.reads), offsets(&site.writes));
    let d_in: usize = site.reads.iter().map(|n| interfaces[*n].width()).sum();
    let d_out: usize = site.writes.iter().map(|n| interfaces[*n].width()).sum();
    let mut parts = Vec::new();
    for (c, component) in components.iter().enumerate() {
        let mut t = Array2::<f64>::zeros((d_out, d_in));
        let mut touched = false;
        for &(w, r, op) in &site.terms {
            if let Some(piece) = component.pieces.get(&program.operators[op].name) {
                let (rows, cols) = (interfaces[site.writes[w]].width(), interfaces[site.reads[r]].width());
                if piece.dim() != (rows, cols) {
                    return Err(format!("{}: a {:?} part of a {rows}×{cols} operator", component.name, piece.dim()));
                }
                let mut target = t.slice_mut(s![wo[w]..wo[w] + rows, ro[r]..ro[r] + cols]);
                target += piece;
                touched = true;
            }
        }
        if touched {
            parts.push((c, t));
        }
    }
    Ok(parts)
}

/// The column and row spaces of `t` (orthonormal bases) at its numerical rank.
fn spaces(t: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let d = svd(t.view(), false).map_err(|e| e.to_string())?;
    let rank = d.singular_values.iter().filter(|v| **v > d.band).count();
    Ok((d.u.slice(s![.., ..rank]).to_owned(), d.vt.slice(s![..rank, ..]).t().to_owned()))
}

/// One site's mechanism check.
#[derive(Clone, Debug, serde::Serialize)]
pub struct SiteMechanism {
    pub site: String,
    pub components: Vec<String>,
    /// `‖M_b‖²`-weighted mean share of each block in its assigned component's subspace.
    pub purity: f64,
    /// Per component touching the site: the share of its part carried by its assigned blocks.
    pub coverage: Vec<f64>,
    /// Blocks assigned per component, and the site's blocks.
    pub assigned: Vec<usize>,
    pub blocks: usize,
    /// `‖W − Σ_c T_c‖ / ‖W‖`: how much of the site the truth leaves unnamed.
    pub unnamed: f64,
    /// Per component with active prompts: the F1 of the prompts its assigned blocks run on.
    pub activity: Vec<Option<f64>>,
}

/// The mechanism checks of an explanation against a case's truth (module note).
#[derive(Clone, Debug, serde::Serialize)]
pub struct Mechanism {
    pub sites: Vec<SiteMechanism>,
    /// Per component with active prompts and no weights: the best single block's F1 and which.
    pub rules: Vec<(String, f64, String)>,
    /// Per attention truth: the explanation's mean weight on the targets, and the model's.
    pub attention: Vec<(String, f64, f64)>,
    pub recovered: bool,
}

fn f1(on: &[bool], active: &[bool]) -> f64 {
    let hits = on.iter().zip(active).filter(|(a, b)| **a && **b).count() as f64;
    let (predicted, actual) = (on.iter().filter(|a| **a).count() as f64, active.iter().filter(|a| **a).count() as f64);
    if predicted + actual == 0.0 { 1.0 } else { 2.0 * hits / (predicted + actual) }
}

/// The mechanism checks (module note) of `explained` on `case`.
pub fn mechanism(case: &Case, explained: &Explained) -> Result<Mechanism, String> {
    let model = &case.imported.program;
    let truth = &case.truth;
    let probe: Vec<&[f64]> = truth.probe.iter().map(Vec::as_slice).collect();
    let length = probe.first().map_or(0, |p| p.len());
    let (probe_trace, sets) = if probe.is_empty() {
        (None, Vec::new())
    } else {
        let (trace, sets) = explained.run_with_sets(&case.family(&probe)?, &[])?;
        (Some(trace), sets)
    };
    // A block runs on a probe prompt when it runs at one of its readout rows.
    let rows_of = |p: usize| -> Vec<usize> { if case.per_position() { case.readouts.iter().map(|r| p * length + r).collect() } else { vec![p] } };
    let runs = |set: &Array2<f64>, blocks: &dyn Fn(usize) -> bool| -> Vec<bool> {
        (0..probe.len()).map(|p| rows_of(p).iter().any(|r| (0..set.ncols()).any(|b| blocks(b) && set[[*r, b]] != 0.0))).collect()
    };
    let indicator = |rows: &[usize]| -> Vec<bool> {
        let mut active = vec![false; probe.len()];
        for r in rows.iter().filter(|r| **r < probe.len()) {
            active[*r] = true;
        }
        active
    };
    let mut sites = Vec::new();
    let mut ok = true;
    for (k, fitted) in explained.explanation.sites.iter().enumerate() {
        let parts = site_parts(model, &fitted.site, &truth.components)?;
        if parts.is_empty() {
            continue;
        }
        let projectors = parts.iter().map(|(_, t)| spaces(t)).collect::<Result<Vec<_>, String>>()?;
        let mut start = 0;
        let (mut weighted, mut mass) = (0.0, 0.0);
        let mut sums: Vec<Array2<f64>> = parts.iter().map(|(_, t)| Array2::zeros(t.dim())).collect();
        let mut assigned = vec![0usize; parts.len()];
        let mut owner = Vec::with_capacity(fitted.ranks.len());
        for &rank in &fitted.ranks {
            let m = fitted.library.u.slice(s![start..start + rank, ..]).t().dot(&fitted.library.v.slice(s![start..start + rank, ..]));
            start += rank;
            let norm = m.iter().map(|v| v * v).sum::<f64>();
            let shares: Vec<f64> = projectors.iter().map(|(u, v)| u.t().dot(&m).dot(v).iter().map(|x| x * x).sum::<f64>() / norm.max(f64::MIN_POSITIVE)).collect();
            let best = (0..shares.len()).fold(0, |b, i| if shares[i] > shares[b] { i } else { b });
            weighted += norm * shares[best];
            mass += norm;
            sums[best] += &m;
            assigned[best] += 1;
            owner.push(best);
        }
        let purity = if mass > 0.0 { weighted / mass } else { 0.0 };
        let coverage: Vec<f64> = parts.iter().zip(&sums).map(|((_, t), a)| (t * a).sum() / t.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE)).collect();
        let named = parts.iter().fold(Array2::<f64>::zeros(fitted.w.dim()), |acc, (_, t)| acc + t);
        let unnamed = (&fitted.w - &named).iter().map(|x| x * x).sum::<f64>().sqrt() / fitted.w.iter().map(|x| x * x).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        let activity: Vec<Option<f64>> = parts
            .iter()
            .enumerate()
            .map(|(i, (c, _))| {
                let active = indicator(truth.components[*c].active.as_ref()?);
                let on = runs(sets.get(k)?, &|b| owner[b] == i);
                Some(f1(&on, &active))
            })
            .collect();
        ok &= purity > 0.5 && coverage.iter().all(|c| *c > 0.5);
        sites.push(SiteMechanism {
            site: fitted.site.name.clone(),
            components: parts.iter().map(|(c, _)| truth.components[*c].name.clone()).collect(),
            purity,
            coverage,
            assigned,
            blocks: fitted.ranks.len(),
            unnamed,
            activity,
        });
    }
    // A component's activity is its best site's.
    for component in truth.components.iter().filter(|c| c.active.is_some() && !c.pieces.is_empty()) {
        let best = sites
            .iter()
            .filter_map(|s| s.components.iter().position(|n| *n == component.name).and_then(|i| s.activity[i]))
            .fold(f64::NEG_INFINITY, f64::max);
        ok &= best > 0.5;
    }
    let mut rules = Vec::new();
    for component in truth.components.iter().filter(|c| c.pieces.is_empty()) {
        let Some(active_rows) = &component.active else { continue };
        let active = indicator(active_rows);
        let mut best = (f64::NEG_INFINITY, String::new());
        for (k, set) in sets.iter().enumerate() {
            for b in 0..set.ncols() {
                let score = f1(&runs(set, &|c| c == b), &active);
                if score > best.0 {
                    best = (score, format!("{} block {b}", explained.explanation.sites[k].site.name));
                }
            }
        }
        ok &= best.0 > 0.5;
        rules.push((component.name.clone(), best.0, best.1));
    }
    let mut attention = Vec::new();
    if let Some(trace) = &probe_trace {
        let native = Native(model);
        let native_trace = native.run(&case.family(&probe)?, &[])?;
        for a in &truth.attention {
            let ours = case.patterns(trace, explained, a.layer, a.head)?;
            let theirs = case.patterns(&native_trace, &native, a.layer, a.head)?;
            let (mut sum_ours, mut sum_theirs, mut count) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (p, targets) in a.targets.iter().enumerate() {
                for &(query, key) in targets {
                    sum_ours += ours[p][[query, key]];
                    sum_theirs += theirs[p][[query, key]];
                    count += 1.0;
                }
            }
            let mean = sum_ours / count.max(1.0);
            ok &= mean > 0.5;
            attention.push((a.name.clone(), mean, sum_theirs / count.max(1.0)));
        }
    }
    Ok(Mechanism { sites, rules, attention, recovered: ok })
}

/// The case directories under `root` (every directory with a `questions.json`), by name.
pub fn cases(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.join("questions.json").is_file())
        .collect();
    found.sort();
    Ok(found)
}
