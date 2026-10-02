//! Identifiability of a decomposition (#2951): is the returned program the only one the code
//! allows, and if not, how do the others differ?
//!
//! The candidates are the search's ties ([`super::engine::Tie`]): certified programs whose total
//! the code cannot separate from the returned one. Each is put in one of four relations to it:
//!
//! * **(a) gauge** — the same structure bits and the same function on the family (every output
//!   row equal within both programs' bands, up to the softmax shift): the two differ by an exact
//!   symmetry. The class itself is reported as the program's gauge generators ([`gauge`]).
//! * **(b) abstraction** — the same function on the family, reached by exact rewrites, with other
//!   structure bits: one states the other at another level of detail.
//! * **(c) redundant** — a different function within the certified band that no intervention
//!   tried tells apart from the returned program: the model's own implementation is redundant
//!   there, as far as the interventions see.
//! * **(d) distinct** — a different hypothesis: an intervention, executed natively on the model,
//!   that resolves the tie: under it one program's total code for the intervened model's
//!   behaviour is proven shorter than the other's. The witness is reported with the side the
//!   model takes.
//!
//! The program is *identified* when no tie is distinct in the tie's favour: every alternative
//! the code allowed on the family is either the same program or refuted by the model under an
//! intervention. The interventions are unit ablations:
//! every group of every pointwise layer set to the zero law, in the model and in both programs at
//! once (layers matched by order, as the search keeps them); the ablation with the largest
//! screened disagreement is certified. The claim is relative to the ties the search certified and
//! to these interventions; it is not a proof over all programs. What is proven is the factor
//! fit's own uniqueness: the finest rule partition of a layer's units is unique, and the bases
//! realising it differ only by the gauge reported here (`factors`, "When the rules are
//! identified").

use super::contract::{Contract, ProgramScore};
use super::engine::{Decomposition, Edit, EngineError, apply_edit};
use super::operator_program::{LabelKind, Law, Node, OperatorProgram};
use std::fmt;

/// How a tie relates to the returned program (module note).
#[derive(Clone, Debug, PartialEq)]
pub enum Relation {
    Gauge,
    Abstraction,
    Redundant,
    Distinct(Witness),
}

/// An intervention that resolves a tie, executed natively.
#[derive(Clone, Debug, PartialEq)]
pub struct Witness {
    pub intervention: String,
    /// The intervened total intervals `[lower, upper]` of the returned program and the tie.
    pub returned_bits: (f64, f64),
    pub tie_bits: (f64, f64),
    /// Whether the intervened model favours the tie (the returned program is refuted).
    pub favours_tie: bool,
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gauge => write!(f, "gauge"),
            Self::Abstraction => write!(f, "abstraction"),
            Self::Redundant => write!(f, "redundant"),
            Self::Distinct(w) => write!(
                f,
                "distinct under {}: returned [{:.1}, {:.1}] bits, tie [{:.1}, {:.1}] bits; the model favours the {}",
                w.intervention,
                w.returned_bits.0,
                w.returned_bits.1,
                w.tie_bits.0,
                w.tie_bits.1,
                if w.favours_tie { "tie" } else { "returned program" }
            ),
        }
    }
}

/// One tie and its relation.
#[derive(Clone, Debug)]
pub struct Alternative {
    pub description: String,
    pub structure_bits: u64,
    pub total: f64,
    pub relation: Relation,
}

/// The identifiability report.
#[derive(Clone, Debug)]
pub struct Identification {
    /// The exact gauge generators of the returned program: its equivalence class under (a).
    pub gauge: Vec<String>,
    pub alternatives: Vec<Alternative>,
    /// No tie is distinct in its own favour.
    pub identified: bool,
}

/// The exact symmetries of `program`'s execution: the generators of its gauge class.
pub fn gauge(program: &OperatorProgram) -> Vec<String> {
    let mut out = Vec::new();
    let interfaces = match program.interfaces() {
        Ok(interfaces) => interfaces,
        Err(error) => return vec![format!("no interfaces: {error}")],
    };
    for (index, node) in program.nodes.iter().enumerate() {
        match node {
            Node::Pointwise { laws, .. } => {
                let relu = laws.iter().filter(|l| **l == Law::Relu).count();
                out.push(format!(
                    "node {index}: permutations of its {} units within each law; a positive scale on each of its {relu} ReLU units (read row × s, write column / s)",
                    laws.len()
                ));
            }
            Node::Affine { .. } if interfaces[index].groups().iter().all(|g| g.label.kind == LabelKind::Factor) => {
                out.push(format!(
                    "node {index}: GL({}) of its factor coordinates, fixed by the shortest coefficient code up to permutation, sign and scale",
                    interfaces[index].width()
                ));
            }
            Node::Readout { .. } => out.push(format!("node {index}: a constant added to every logit (softmax shift)")),
            Node::RmsNorm { .. } => out.push(format!("node {index}: a positive scale of its input")),
            _ => {}
        }
    }
    out
}

/// Whether two programs compute the same distributions on the family: every row equal within
/// the two bands after removing the row's mean shift.
fn same_function(contract: &Contract, a: &OperatorProgram, b: &OperatorProgram) -> Result<bool, EngineError> {
    let (x, y) = (contract.logits(a)?, contract.logits(b)?);
    for row in 0..x.values.nrows() {
        let (xr, yr) = (x.values.row(row), y.values.row(row));
        let shift = (xr.sum() - yr.sum()) / xr.len() as f64;
        let slack = (x.bands.row(row).iter().fold(0.0_f64, |m, v| m.max(*v)) + y.bands.row(row).iter().fold(0.0_f64, |m, v| m.max(*v))) * 2.0;
        if xr.iter().zip(yr.iter()).any(|(p, q)| (p - q - shift).abs() > slack) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The pointwise nodes of a program, in order.
fn layers(program: &OperatorProgram) -> Vec<(usize, Vec<Law>)> {
    program
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| match n {
            Node::Pointwise { laws, .. } => Some((i, laws.clone())),
            _ => None,
        })
        .collect()
}

/// `program` with group `group` of its `layer`-th pointwise node at the zero law.
fn ablate(program: &OperatorProgram, layer: usize, group: usize) -> Result<Option<OperatorProgram>, EngineError> {
    let Some((node, laws)) = layers(program).into_iter().nth(layer) else { return Ok(None) };
    if group >= laws.len() {
        return Ok(None);
    }
    let mut laws = laws;
    laws[group] = Law::Zero;
    let mut out = program.clone();
    apply_edit(&mut out, &Edit::Laws { node, laws, blocks: Vec::new() })?;
    Ok(Some(out))
}

fn interval(score: &ProgramScore) -> (f64, f64) {
    (score.total_lower(), score.total_upper())
}

/// The unit ablation on which `returned` and `tie` disagree most about the natively ablated model,
/// certified (module note); `None` when no ablation separates them.
fn witness(model: &OperatorProgram, contract: &Contract, returned: &OperatorProgram, tie: &OperatorProgram) -> Result<Option<Witness>, EngineError> {
    let model_layers = layers(model);
    let (r_layers, t_layers) = (layers(returned), layers(tie));
    let mut best: Option<(f64, usize, usize)> = None;
    for (layer, (_, laws)) in model_layers.iter().enumerate() {
        let comparable = |ls: &Vec<(usize, Vec<Law>)>| ls.get(layer).is_some_and(|(_, l)| l.len() == laws.len());
        if !comparable(&r_layers) || !comparable(&t_layers) {
            continue;
        }
        for group in 0..laws.len() {
            if laws[group] == Law::Zero {
                continue;
            }
            let (Some(r), Some(t)) = (ablate(returned, layer, group)?, ablate(tie, layer, group)?) else {
                continue;
            };
            // Screen without bands: the disagreement of the two programs' distributions.
            let execute = |p: &OperatorProgram| -> Result<ndarray::Array2<f64>, EngineError> {
                Ok(contract.distributions(&p.execute(&contract.family, false)?.values[p.output])?)
            };
            let (r_out, t_out) = (execute(&r)?, execute(&t)?);
            let gap = r_out.iter().zip(t_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            if best.is_none_or(|(g, _, _)| gap > g) {
                best = Some((gap, layer, group));
            }
        }
    }
    let Some((_, layer, group)) = best else { return Ok(None) };
    let (Some(m), Some(r), Some(t)) = (ablate(model, layer, group)?, ablate(returned, layer, group)?, ablate(tie, layer, group)?) else {
        return Ok(None);
    };
    let reference = contract.logits(&m)?;
    let (rs, ts) = (contract.score(&r, &reference)?, contract.score(&t, &reference)?);
    let (ri, ti) = (interval(&rs), interval(&ts));
    Ok((ri.1 < ti.0 || ti.1 < ri.0).then(|| Witness {
        intervention: format!("unit {group} of pointwise layer {layer} ablated (zero law) in the model and both programs"),
        returned_bits: ri,
        tie_bits: ti,
        favours_tie: ti.1 < ri.0,
    }))
}

/// Classify every tie of `decomposition` (module note).
pub fn identify(model: &OperatorProgram, contract: &Contract, decomposition: &Decomposition) -> Result<Identification, EngineError> {
    let returned = &decomposition.program;
    let mut alternatives = Vec::new();
    for tie in &decomposition.ties {
        let relation = if same_function(contract, returned, &tie.program)? {
            if tie.score.structure_bits == decomposition.score.structure_bits { Relation::Gauge } else { Relation::Abstraction }
        } else {
            match witness(model, contract, returned, &tie.program)? {
                Some(w) => Relation::Distinct(w),
                None => Relation::Redundant,
            }
        };
        alternatives.push(Alternative {
            description: tie.description.clone(),
            structure_bits: tie.score.structure_bits,
            total: tie.score.total(),
            relation,
        });
    }
    let identified = !alternatives.iter().any(|a| matches!(&a.relation, Relation::Distinct(w) if w.favours_tie));
    Ok(Identification { gauge: gauge(returned), alternatives, identified })
}
