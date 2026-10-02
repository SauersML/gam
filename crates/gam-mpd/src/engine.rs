//! Program decomposition by two-part code (#2951): the operator program whose decoded message plus
//! the excess code length of the model's behaviour under it is shortest.
//!
//! ```text
//! decompose(model, contract, library, budget) -> Decomposition { program, score, stop }
//! ```
//!
//! `model` is the native network as an [`OperatorProgram`] over the contract's declarations; its
//! banded execution is the reference, and its own two-part code (its message, and no excess) is
//! where the search starts. A candidate `P` is scored by `L(P) + Σ_rows KL(model ‖ P)/ln 2`
//! (`contract::ProgramScore`); no tolerance is declared. `library` is a list of [`Primitive`]s.
//! Each proposes candidate replacements for part of the current program: a [`Proposal`] carries its
//! [`Edit`], its exactness (an exact rewrite names its derivation) and its kind. The primitive
//! claims neither the saving nor the effect: the engine measures both.
//!
//! # One round
//!
//! 1. Every primitive proposes on the current program `P`, whose certified score and logits are
//!    cached.
//! 2. Each proposal's program bits are computed exactly (its decoded message length).
//! 3. Each proposal is screened: its logits on the family (for a local edit only what the edit
//!    changes is propagated, `OperatorProgram::execute_incremental`) and the resulting data bits.
//!    A structural proposal whose screened total is not below the current total is retried, when the budget
//!    allows, as one compound move with the readout's reals refitted (`refit::refit_readout`).
//!    Screening certifies nothing; it ranks.
//! 4. The screened proposals are ranked by total saving. A structural proposal (one that replaces
//!    the program) is tried alone; the compatible local edits are tried together.
//! 5. The candidate is certified: encoded, decoded, executed with bands on the whole family and
//!    scored (`contract::Contract::score`). It is accepted only when its total code length is
//!    proven shorter: its upper end below the current program's lower end. A refused batch is
//!    halved; a refused single proposal is not proposed again.
//!
//! The search ends when no proposal passes, or when the declared [`Budget`] is spent; in both cases
//! the returned program is the last certified one, and its maximal row KL and argmax agreement are
//! reported as outputs.

use super::contract::{Contract, ContractError, FamilyKind, ProgramScore};
use super::fit::ProposalKind;
use super::operator_program::{
    FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram, ProgramError, Trace,
    round_to_lattice,
};
use super::refit::{RefitSearch, refit_readout};
use super::precision::DeclaredPrecision;
use super::secant::BandedMatrix;
use ndarray::{Array2, s};
use std::collections::BTreeSet;
use std::fmt;

/// Whether a proposal is an algebraic identity or changes the computed function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Exactness {
    /// Exact in exact arithmetic; the derivation is stated. Rounding still moves the executed
    /// value, so it is certified like any other proposal.
    Exact { derivation: String },
    Approximate,
}

/// One block of one operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockRef {
    pub operator: usize,
    pub row: usize,
    pub col: usize,
}

/// A change of the program.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// Remove present blocks.
    DropBlocks { blocks: Vec<BlockRef> },
    /// Move an operator's reals to another lattice (coarsened when the reals' range needs it,
    /// `DeclaredPrecision::within_range`).
    Precision { operator: usize, precision: DeclaredPrecision },
    /// Replace a pointwise node's laws, and remove the blocks the new laws make unread.
    Laws { node: usize, laws: Vec<Law>, blocks: Vec<BlockRef> },
    /// Replace the program.
    Program(Box<OperatorProgram>),
}

impl Edit {
    fn is_local(&self) -> bool {
        !matches!(self, Self::Program(_))
    }
}

/// A candidate replacement for part of the current program.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub primitive: &'static str,
    pub kind: ProposalKind,
    pub exactness: Exactness,
    /// A stable description, also the key under which a refused proposal is not proposed again.
    pub description: String,
    pub edit: Edit,
}

/// What a primitive sees.
pub struct SearchContext<'a> {
    /// The current program.
    pub program: &'a OperatorProgram,
    pub contract: &'a Contract,
    /// The current program's unbanded trace on the family.
    pub trace: &'a Trace,
    pub interfaces: &'a [Interface],
    /// The search level: `0` coarse proposals, higher levels finer ones (see [`Primitive`]).
    pub level: usize,
}

/// A source of proposals.
///
/// A primitive proposes at the levels it declares: [`Primitive::levels`] is the number of levels
/// it has, finest last. The engine asks every primitive for level 0 and moves to the next level
/// only when no proposal at the current level passes; it returns to level 0 after an acceptance.
/// Levels order the search from coarse to fine; they never exclude a proposal.
pub trait Primitive {
    fn name(&self) -> &'static str;

    fn levels(&self) -> usize {
        1
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError>;
}

/// The declared search budget: screened proposals and certified evaluations, and whether a
/// proposal whose screened total is not shorter is retried with the readout's reals refitted
/// (`refit::refit_readout`) as one compound move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub screenings: u64,
    pub certifications: u64,
    pub refit: Option<RefitSearch>,
}

impl Default for Budget {
    /// The budget every driver runs: about a million screenings, four thousand certifications,
    /// and the readout refit.
    fn default() -> Self {
        Self { screenings: 1 << 20, certifications: 1 << 12, refit: Some(RefitSearch { newton_steps: 8, conjugate_gradient_steps: 16 }) }
    }
}

/// The library every driver runs: the exact rewrites, the restrictions, the curvature
/// precisions, law substitution and the fitted shared factors. Nothing in it names a task.
pub fn library() -> Vec<Box<dyn Primitive>> {
    use super::derivatives::CurvaturePrecision;
    use super::factors::Factors;
    use super::operator_rewrites::{
        BilinearConstantSide, CenterLogits, ComposeAffine, DropKeyBias, FoldConstants, PlaneBasis, PushThroughMix, StackTerms,
    };
    vec![
        Box::new(FoldConstants),
        Box::new(BilinearConstantSide),
        Box::new(ComposeAffine),
        Box::new(PushThroughMix),
        Box::new(PlaneBasis),
        Box::new(DropBlocks),
        Box::new(Coarsen),
        Box::new(DeadUnits),
        Box::new(StackTerms),
        Box::new(CenterLogits),
        Box::new(DropKeyBias),
        Box::new(Factors),
        Box::new(LawSubstitution),
        Box::new(CurvaturePrecision { probes: 4 }),
    ]
}

/// A logger writing the engine's progress (`log::info!`: each accepted move) to standard error,
/// for drivers; installed once, later calls are no-ops.
pub fn log_to_stderr() {
    struct Stderr;
    impl log::Log for Stderr {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Info
        }
        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                eprintln!("{}", record.args());
            }
        }
        fn flush(&self) {}
    }
    static LOGGER: Stderr = Stderr;
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
}

/// Why the search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// No proposal at any level shortened the certified total.
    Converged,
    ScreeningBudget,
    CertificationBudget,
}

/// The result of [`decompose`]: the program, its certified score, why it stopped, and the
/// structure function the search traced.
#[derive(Clone, Debug)]
pub struct Decomposition {
    pub program: OperatorProgram,
    pub score: ProgramScore,
    pub stop: Stop,
    /// The Pareto front of every certified program the search scored: structure bits against
    /// the rest of the code (precision, explanations and data), ascending in structure.
    pub curve: Vec<CurvePoint>,
    /// Certified programs the code cannot tell from the returned one: their totals' certified
    /// intervals overlap it. Each is an edit of a program the search held, or another start's
    /// result (`identify` classifies them).
    pub ties: Vec<Tie>,
}

/// A certified program whose total is not proven longer than the returned program's.
#[derive(Clone, Debug)]
pub struct Tie {
    pub program: OperatorProgram,
    pub score: ProgramScore,
    pub description: String,
    /// Whether every edit that produced it from the held program is exact in exact arithmetic.
    pub exact: bool,
}

/// One certified program on the structure function.
#[derive(Clone, Debug, PartialEq)]
pub struct CurvePoint {
    pub structure_bits: u64,
    /// Precision, explanation and data bits at the computed values.
    pub rest_bits: f64,
    pub description: String,
}

impl CurvePoint {
    fn of(score: &ProgramScore, description: String) -> Self {
        Self { structure_bits: score.structure_bits, rest_bits: score.total() - score.structure_bits as f64, description }
    }
}

/// The points no other point beats in both structure and the rest, ascending in structure.
fn pareto(mut points: Vec<CurvePoint>) -> Vec<CurvePoint> {
    points.sort_by(|a, b| a.structure_bits.cmp(&b.structure_bits).then_with(|| a.rest_bits.total_cmp(&b.rest_bits)));
    let mut front: Vec<CurvePoint> = Vec::new();
    for point in points {
        if front.last().is_none_or(|last| point.rest_bits < last.rest_bits) {
            front.push(point);
        }
    }
    front
}

impl Decomposition {
    /// The knee of the structure function: on the lower convex hull of [`Self::curve`], the first
    /// vertex after which one more bit of structure buys less than one bit of the rest (the
    /// hull's slope reaches `−1`). It is the least-structure program of least total on the
    /// hull, derived from the code alone.
    pub fn knee(&self) -> Option<&CurvePoint> {
        let points = &self.curve;
        let mut hull: Vec<usize> = Vec::new();
        for index in 0..points.len() {
            while hull.len() >= 2 {
                let (a, b) = (&points[hull[hull.len() - 2]], &points[hull[hull.len() - 1]]);
                let c = &points[index];
                let cross = (b.structure_bits as f64 - a.structure_bits as f64) * (c.rest_bits - a.rest_bits)
                    - (b.rest_bits - a.rest_bits) * (c.structure_bits as f64 - a.structure_bits as f64);
                if cross <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(index);
        }
        for pair in hull.windows(2) {
            let (a, b) = (&points[pair[0]], &points[pair[1]]);
            let slope = (b.rest_bits - a.rest_bits) / (b.structure_bits as f64 - a.structure_bits as f64);
            if slope >= -1.0 {
                return Some(a);
            }
        }
        hull.last().map(|&i| &points[i])
    }
}

/// A refused decomposition.
#[derive(Debug)]
pub enum EngineError {
    Program(ProgramError),
    Contract(ContractError),
    /// A start program whose score is unresolved: its data bits have no upper end.
    UnresolvedStart(String),
    Primitive(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(f, "decompose: {error}"),
            Self::Contract(error) => write!(f, "decompose: {error}"),
            Self::UnresolvedStart(message) => write!(f, "decompose: the start program's score is unresolved: {message}"),
            Self::Primitive(message) => write!(f, "decompose: primitive: {message}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<ProgramError> for EngineError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

impl From<ContractError> for EngineError {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}

/// Apply `edit` to `program`.
pub fn apply_edit(program: &mut OperatorProgram, edit: &Edit) -> Result<(), EngineError> {
    match edit {
        Edit::DropBlocks { blocks } => drop_blocks(program, blocks),
        Edit::Precision { operator, precision } => {
            let op = program
                .operators
                .get_mut(*operator)
                .ok_or(EngineError::Program(ProgramError::Reference { what: "operator", index: *operator }))?;
            // The target lattice is derived from the operator's range, so a proposed step never
            // overflows the lattice code.
            let precision = &precision.within_range(op.largest_real());
            match &mut op.body {
                OperatorBody::Dense { values, present, precision: current } => {
                    for ((r, c), &keep) in present.indexed_iter() {
                        if keep {
                            for value in values.slice_mut(s![op.rows.range(r), op.cols.range(c)]).iter_mut() {
                                *value = round_to_lattice(*value, *precision)?;
                            }
                        }
                    }
                    *current = *precision;
                }
                OperatorBody::LowRank { left, right, precision: current } => {
                    for value in left.iter_mut().chain(right.iter_mut()) {
                        *value = round_to_lattice(*value, *precision)?;
                    }
                    *current = *precision;
                }
                OperatorBody::Identity => {
                    return Err(EngineError::Primitive(format!("operator {} has no reals to re-precise", op.name)));
                }
            }
            Ok(())
        }
        Edit::Laws { node, laws, blocks } => {
            match program.nodes.get_mut(*node) {
                Some(Node::Pointwise { laws: current, .. }) if current.len() == laws.len() => current.clone_from(laws),
                _ => return Err(EngineError::Primitive(format!("node {node} is not a pointwise node of {} groups", laws.len()))),
            }
            drop_blocks(program, blocks)
        }
        Edit::Program(candidate) => {
            *program = (**candidate).clone();
            Ok(())
        }
    }
}

fn drop_blocks(program: &mut OperatorProgram, blocks: &[BlockRef]) -> Result<(), EngineError> {
    for block in blocks {
        let op = program
            .operators
            .get_mut(block.operator)
            .ok_or(EngineError::Program(ProgramError::Reference { what: "operator", index: block.operator }))?;
        let (rows, cols) = (op.rows.range(block.row), op.cols.range(block.col));
        let OperatorBody::Dense { values, present, .. } = &mut op.body else {
            return Err(EngineError::Primitive(format!("operator {} has no blocks", op.name)));
        };
        let keep = present
            .get_mut((block.row, block.col))
            .ok_or_else(|| EngineError::Primitive(format!("block {block:?} is outside its operator")))?;
        *keep = false;
        values.slice_mut(s![rows, cols]).fill(0.0);
    }
    Ok(())
}

/// The reference distributions, cached.
struct Reference {
    banded: BandedMatrix,
    log_probabilities: Array2<f64>,
}

fn log_softmax_rows(logits: &Array2<f64>) -> Array2<f64> {
    let mut out = logits.clone();
    for mut row in out.outer_iter_mut() {
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
        row.mapv_inplace(|v| v - lse);
    }
    out
}

/// `Σ_rows KL(model ‖ candidate) / ln 2`, unbanded: a screen, not a certificate.
fn screened_data_bits(reference: &Reference, candidate: &Array2<f64>, observations: u64) -> f64 {
    let log_q = log_softmax_rows(candidate);
    let mut total = 0.0;
    for (lp, lq) in reference.log_probabilities.outer_iter().zip(log_q.outer_iter()) {
        total += lp.iter().zip(lq.iter()).map(|(a, b)| a.exp() * (a - b)).sum::<f64>();
    }
    observations as f64 * total / std::f64::consts::LN_2
}

/// A screened proposal.
struct Screened {
    proposal: Proposal,
    /// The screened total saving in bits.
    saving: f64,
}

/// Two local edits that may be applied together.
fn compatible(a: &Edit, b: &Edit) -> bool {
    match (a, b) {
        (Edit::Precision { operator: x, .. }, Edit::Precision { operator: y, .. }) => x != y,
        (Edit::Laws { node: x, .. }, Edit::Laws { node: y, .. }) => x != y,
        (Edit::DropBlocks { blocks: x }, Edit::DropBlocks { blocks: y }) => x.iter().all(|b| !y.contains(b)),
        (Edit::Precision { operator, .. }, Edit::DropBlocks { blocks })
        | (Edit::DropBlocks { blocks }, Edit::Precision { operator, .. }) => blocks.iter().all(|b| b.operator != *operator),
        (Edit::Precision { operator, .. }, Edit::Laws { blocks, .. })
        | (Edit::Laws { blocks, .. }, Edit::Precision { operator, .. }) => blocks.iter().all(|b| b.operator != *operator),
        (Edit::Laws { blocks: x, .. }, Edit::DropBlocks { blocks: y })
        | (Edit::DropBlocks { blocks: y }, Edit::Laws { blocks: x, .. }) => x.iter().all(|b| !y.contains(b)),
        _ => false,
    }
}

/// `candidate` with the dense operators of the affine node its readout reads refitted, when its
/// output is such a readout.
fn refit_after(
    candidate: &OperatorProgram,
    contract: &Contract,
    reference: &Reference,
    search: RefitSearch,
) -> Result<Option<OperatorProgram>, EngineError> {
    if contract.readouts != 1 {
        return Ok(None);
    }
    let Node::Readout { input, .. } = candidate.nodes[candidate.output] else { return Ok(None) };
    let Node::Affine { terms, bias } = &candidate.nodes[input] else { return Ok(None) };
    let operators: Vec<usize> = terms
        .iter()
        .map(|(_, op)| *op)
        .chain(bias.iter().copied())
        .filter(|op| matches!(candidate.operators[*op].body, OperatorBody::Dense { .. }))
        .collect();
    if operators.is_empty() {
        return Ok(None);
    }
    let target = reference.log_probabilities.mapv(f64::exp);
    Ok(Some(refit_readout(candidate, &contract.family, &target, &operators, search)?))
}

/// Decompose `model` under `contract` with `library`, within `budget`.
pub fn decompose(
    model: &OperatorProgram,
    contract: &Contract,
    library: &[Box<dyn Primitive>],
    budget: &Budget,
) -> Result<Decomposition, EngineError> {
    decompose_from(model, model, contract, library, budget)
}

/// [`decompose`] with `start` (a program over the model's declarations, for instance the result of
/// a search on less behaviour) joining the start set. The start set is always `{model, start}`: the
/// search runs from each, and the shortest certified result is returned, so a warm start can never
/// leave the result longer than the model's own search would.
pub fn decompose_from(
    model: &OperatorProgram,
    start: &OperatorProgram,
    contract: &Contract,
    library: &[Box<dyn Primitive>],
    budget: &Budget,
) -> Result<Decomposition, EngineError> {
    contract.validate()?;
    let banded = contract.logits(model)?;
    decompose_with_reference(&banded, &[model, start], contract, library, budget)
}

/// [`decompose_from`] with the model's banded logits on the family given: `reference` must be
/// `contract.logits(model)` (restricted to the family's rows when the family is a selection), so a
/// caller scoring many families against one model runs the model once. `starts` is the start set
/// (the model among them): the search runs from each distinct start, shortest first, each within
/// `budget`. The search itself only moves to proven-shorter programs, so no result is longer than
/// its start.
///
/// Of the results, those not proven longer than the shortest (their certified totals overlap) are
/// equally short as far as the code can tell; the one with the least structure bits is returned:
/// the knee of the structure function, where more structure no longer buys a proven-shorter total.
pub fn decompose_with_reference(
    reference: &BandedMatrix,
    starts: &[&OperatorProgram],
    contract: &Contract,
    library: &[Box<dyn Primitive>],
    budget: &Budget,
) -> Result<Decomposition, EngineError> {
    contract.validate()?;
    let expected = contract.family.rows * contract.readouts;
    if reference.values.nrows() != expected {
        return Err(EngineError::Contract(ContractError::Declaration(format!(
            "{} reference rows for {} inputs at {} readouts",
            reference.values.nrows(),
            contract.family.rows,
            contract.readouts
        ))));
    }
    let reference = Reference { log_probabilities: log_softmax_rows(&reference.values), banded: reference.clone() };
    let mut distinct: Vec<&OperatorProgram> = Vec::new();
    for start in starts {
        if !distinct.iter().any(|seen| *seen == *start) {
            distinct.push(start);
        }
    }
    let mut scored = Vec::with_capacity(distinct.len());
    for start in distinct {
        let score = contract.score(start, &reference.banded)?;
        if !score.total_upper().is_finite() {
            return Err(EngineError::UnresolvedStart(format!("{:?}", score.evaluation.total_kl)));
        }
        scored.push((start, score));
    }
    scored.sort_by(|a, b| a.1.total_upper().total_cmp(&b.1.total_upper()));
    let mut results = Vec::with_capacity(scored.len());
    for (start, score) in scored {
        results.push(search(&reference, start.clone(), score, contract, library, budget)?);
    }
    let curve = pareto(results.iter().flat_map(|r| r.curve.iter().cloned()).collect());
    let shortest = results.iter().map(|r| r.score.total_upper()).fold(f64::INFINITY, f64::min);
    let chosen_index = results
        .iter()
        .enumerate()
        .filter(|(_, r)| r.score.total_lower() <= shortest)
        .min_by(|(_, a), (_, b)| {
            a.score
                .structure_bits
                .cmp(&b.score.structure_bits)
                .then_with(|| a.score.total_upper().total_cmp(&b.score.total_upper()))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| EngineError::UnresolvedStart("an empty start set".to_string()))?;
    let mut chosen = results.swap_remove(chosen_index);
    // The other starts' results and ties that the code cannot separate from the chosen program.
    let mut ties = std::mem::take(&mut chosen.ties);
    for other in results {
        if !chosen.score.proven_shorter_than(&other.score) && other.program != chosen.program {
            ties.push(Tie { program: other.program, score: other.score, description: "another start's result".to_string(), exact: false });
        }
        ties.extend(other.ties);
    }
    ties.retain(|tie| !chosen.score.proven_shorter_than(&tie.score) && tie.program != chosen.program);
    chosen.ties = ties;
    chosen.curve = curve;
    Ok(chosen)
}

/// The search from one start program whose certified score is `start_score`.
fn search(
    reference: &Reference,
    start: OperatorProgram,
    start_score: ProgramScore,
    contract: &Contract,
    library: &[Box<dyn Primitive>],
    budget: &Budget,
) -> Result<Decomposition, EngineError> {
    let mut program = start;
    let mut curve = vec![CurvePoint::of(&start_score, "start".to_string())];
    let mut ties: Vec<Tie> = Vec::new();
    let mut current = start_score;
    let mut refused: BTreeSet<String> = BTreeSet::new();
    // Proposals whose compound move with a refit has been screened once and did not pay.
    let mut refit_tried: BTreeSet<String> = BTreeSet::new();
    let (mut screenings, mut certifications) = (0u64, 0u64);
    let levels = library.iter().map(|primitive| primitive.levels()).max().unwrap_or(1);
    let mut level = 0;
    let stop = 'search: loop {
        let trace = program.execute(&contract.family, false)?;
        let interfaces = program.interfaces()?;
        let context = SearchContext { program: &program, contract, trace: &trace, interfaces: &interfaces, level };
        let mut proposals = Vec::new();
        for primitive in library {
            if level < primitive.levels() {
                proposals.extend(primitive.propose(&context)?);
            }
        }
        let base_total = current.total();
        // Screening does not re-derive the explanations; it charges the current program's.
        let explanation_bits = current.explanation.bits;
        let mut screened: Vec<Screened> = Vec::new();
        for proposal in proposals {
            if refused.contains(&proposal.description) {
                continue;
            }
            let mut candidate = program.clone();
            apply_edit(&mut candidate, &proposal.edit)?;
            let bits = candidate.code_bits()? as f64;
            if screenings >= budget.screenings {
                break 'search Stop::ScreeningBudget;
            }
            screenings += 1;
            let logits = contract.distributions(&if proposal.edit.is_local() {
                candidate.execute_incremental(&contract.family, &program, &trace)?
            } else {
                candidate.execute(&contract.family, false)?.values[candidate.output].clone()
            })?;
            let mut saving = base_total - (bits + explanation_bits + screened_data_bits(&reference, &logits, contract.observations));
            let mut proposal = proposal;
            // The compound move is tried for structural proposals only, once each: restrictions
            // and precisions come in the thousands, and a refit each round would be the search.
            if saving <= 0.0
                && bits < base_total
                && !proposal.edit.is_local()
                && refit_tried.insert(proposal.description.clone())
                && let Some(search) = budget.refit
                && let Some(refit) = refit_after(&candidate, contract, &reference, search)?
            {
                // The compound move: the edit and a refit of the readout's reals, judged together.
                let refit_bits = refit.code_bits()? as f64;
                let refit_logits = contract.distributions(&refit.execute(&contract.family, false)?.values[refit.output])?;
                let refit_saving =
                    base_total - (refit_bits + explanation_bits + screened_data_bits(&reference, &refit_logits, contract.observations));
                if refit_saving > 0.0 {
                    saving = refit_saving;
                    proposal = Proposal {
                        description: format!("{} + refit", proposal.description),
                        exactness: Exactness::Approximate,
                        edit: Edit::Program(Box::new(refit)),
                        ..proposal
                    };
                }
            }
            if saving > 0.0 {
                screened.push(Screened { proposal, saving });
            }
        }
        if screened.is_empty() {
            if level + 1 < levels {
                level += 1;
                continue;
            }
            break Stop::Converged;
        }
        screened.sort_by(|a, b| {
            b.saving.total_cmp(&a.saving).then_with(|| a.proposal.description.cmp(&b.proposal.description))
        });
        let batch: Vec<usize> = if !screened[0].proposal.edit.is_local() {
            vec![0]
        } else {
            let mut chosen: Vec<usize> = Vec::new();
            for (index, item) in screened.iter().enumerate() {
                if item.proposal.edit.is_local()
                    && chosen.iter().all(|&c| compatible(&screened[c].proposal.edit, &item.proposal.edit))
                {
                    chosen.push(index);
                }
            }
            chosen
        };
        let mut size = batch.len();
        let accepted = loop {
            if certifications >= budget.certifications {
                break 'search Stop::CertificationBudget;
            }
            certifications += 1;
            let members = &batch[..size];
            let mut candidate = program.clone();
            for &member in members {
                apply_edit(&mut candidate, &screened[member].proposal.edit)?;
            }
            candidate.prune();
            let score = contract.score(&candidate, &reference.banded)?;
            curve.push(CurvePoint::of(
                &score,
                members.iter().map(|&m| screened[m].proposal.description.as_str()).collect::<Vec<_>>().join("; "),
            ));
            if score.proven_shorter_than(&current) {
                log::info!(
                    "accepted {} proposal(s): {} + {:.1} bits -> {} + {:.1} bits: {:?}",
                    members.len(),
                    current.program_bits,
                    current.data_bits,
                    score.program_bits,
                    score.data_bits,
                    members.iter().map(|&m| screened[m].proposal.description.as_str()).collect::<Vec<_>>()
                );
                break Some((candidate, score));
            }
            if !current.proven_shorter_than(&score) {
                ties.push(Tie {
                    program: candidate,
                    score: score.clone(),
                    description: members.iter().map(|&m| screened[m].proposal.description.as_str()).collect::<Vec<_>>().join("; "),
                    exact: members.iter().all(|&m| matches!(screened[m].proposal.exactness, Exactness::Exact { .. })),
                });
            }
            if size > 1 {
                size /= 2;
            } else {
                refused.insert(screened[members[0]].proposal.description.clone());
                break None;
            }
        };
        if let Some((candidate, score)) = accepted {
            program = candidate;
            current = score;
            ties.retain(|tie| !current.proven_shorter_than(&tie.score));
            level = 0;
        }
    };
    Ok(Decomposition { program, score: current, stop, curve: pareto(curve), ties })
}

/// A decomposition refined against a pool of inputs the family does not hold.
#[derive(Clone, Debug)]
pub struct Refinement {
    pub decomposition: Decomposition,
    /// Pool inputs that refuted a round's program and joined the family.
    pub added: usize,
    pub pool: usize,
    /// The last program's KL on every pool row is within its certified maximum on the family:
    /// the claim survived a search over the whole pool (a search, not a proof beyond the family).
    pub survived: bool,
}

/// Counterexample-guided refinement: decompose; run the program and the model on `pool`; every
/// pool input with a distribution row whose KL exceeds the program's certified maximum KL on the
/// family refutes that the family's worst case covers the pool, and joins the family (as a new unit
/// of a sampled family); decompose again from the current program. At most `rounds` rounds.
pub fn decompose_refined(
    model: &OperatorProgram,
    contract: &Contract,
    library: &[Box<dyn Primitive>],
    budget: &Budget,
    pool: &FamilyInputs,
    rounds: usize,
) -> Result<Refinement, EngineError> {
    let mut contract = contract.clone();
    let mut start = model.clone();
    let mut added = 0;
    let model_pool = contract.distributions(&model.execute(pool, false)?.values[model.output])?;
    let reference = Reference { log_probabilities: log_softmax_rows(&model_pool), banded: BandedMatrix { values: model_pool.clone(), bands: Array2::zeros(model_pool.dim()) } };
    for round in 0..rounds.max(1) {
        let result = decompose_from(model, &start, &contract, library, budget)?;
        let threshold = result.score.evaluation.max_kl.upper_bound().unwrap_or(f64::INFINITY);
        let program_pool = contract.distributions(&result.program.execute(pool, false)?.values[result.program.output])?;
        let log_q = log_softmax_rows(&program_pool);
        let mut refuting: BTreeSet<usize> = BTreeSet::new();
        for (row, (lp, lq)) in reference.log_probabilities.outer_iter().zip(log_q.outer_iter()).enumerate() {
            let kl: f64 = lp.iter().zip(lq.iter()).map(|(a, b)| a.exp() * (a - b)).sum();
            if kl > threshold {
                refuting.insert(row / contract.readouts);
            }
        }
        if refuting.is_empty() || round + 1 == rounds.max(1) {
            return Ok(Refinement { decomposition: result, added, pool: pool.rows, survived: refuting.is_empty() });
        }
        let rows: Vec<usize> = refuting.into_iter().collect();
        added += rows.len();
        let extra = pool.select(&rows);
        if let FamilyKind::Sample { units, .. } = &mut contract.kind {
            let next = units.iter().copied().max().map_or(0, |m| m + 1);
            units.extend((0..extra.rows).map(|i| next + i));
        }
        contract.family = contract.family.append(&extra)?;
        start = result.program;
    }
    Err(EngineError::Primitive("refinement ran no round".to_string()))
}

// ------------------------------------------------------------------------------------ restrictions

/// Remove present blocks: level 0 drops a whole column group of an operator (an input subspace
/// removed from every row it feeds), or a whole row group (an output coordinate no longer
/// written); level 1 drops one row group's reads of one column label tied across the term
/// operators of an affine node that share it (e.g. one unit's read of plane `k` in every head);
/// level 2 drops single blocks.
pub struct DropBlocks;

fn present_blocks(op: &Operator) -> Vec<(usize, usize)> {
    match &op.body {
        OperatorBody::Dense { present, .. } => present.indexed_iter().filter(|(_, k)| **k).map(|(rc, _)| rc).collect(),
        OperatorBody::Identity | OperatorBody::LowRank { .. } => Vec::new(),
    }
}

impl Primitive for DropBlocks {
    fn name(&self) -> &'static str {
        "drop_blocks"
    }

    fn levels(&self) -> usize {
        3
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        let proposal = |description: String, blocks: Vec<BlockRef>| Proposal {
            primitive: "drop_blocks",
            kind: ProposalKind::Reduce,
            exactness: Exactness::Approximate,
            description,
            edit: Edit::DropBlocks { blocks },
        };
        match context.level {
            0 => {
                for (index, op) in program.operators.iter().enumerate() {
                    let blocks = present_blocks(op);
                    for col in 0..op.cols.group_count() {
                        let dropped: Vec<BlockRef> = blocks
                            .iter()
                            .filter(|(_, c)| *c == col)
                            .map(|&(row, col)| BlockRef { operator: index, row, col })
                            .collect();
                        if !dropped.is_empty() {
                            out.push(proposal(format!("drop input {:?} of {}", op.cols.groups()[col].label, op.name), dropped));
                        }
                    }
                    if op.rows.group_count() > 1 {
                        for row in 0..op.rows.group_count() {
                            let dropped: Vec<BlockRef> = blocks
                                .iter()
                                .filter(|(r, _)| *r == row)
                                .map(|&(row, col)| BlockRef { operator: index, row, col })
                                .collect();
                            if !dropped.is_empty() {
                                out.push(proposal(format!("drop output {:?} of {}", op.rows.groups()[row].label, op.name), dropped));
                            }
                        }
                    }
                }
            }
            1 => {
                for (node_index, node) in program.nodes.iter().enumerate() {
                    let Node::Affine { terms, .. } = node else { continue };
                    if terms.len() < 2 {
                        continue;
                    }
                    let mut labels: BTreeSet<(usize, (LabelKind, u32))> = BTreeSet::new();
                    for (_, op) in terms {
                        for (row, col) in present_blocks(&program.operators[*op]) {
                            let label = program.operators[*op].cols.groups()[col].label;
                            labels.insert((row, (label.kind, label.index)));
                        }
                    }
                    for (row, (kind, label_index)) in labels {
                        let mut blocks = Vec::new();
                        for (_, op) in terms {
                            let operator = &program.operators[*op];
                            for (r, c) in present_blocks(operator) {
                                let label = operator.cols.groups()[c].label;
                                if r == row && label.kind == kind && label.index == label_index {
                                    blocks.push(BlockRef { operator: *op, row: r, col: c });
                                }
                            }
                        }
                        if blocks.len() > 1 {
                            out.push(proposal(format!("drop row {row} reads of {kind:?}{label_index} tied in node {node_index}"), blocks));
                        }
                    }
                }
            }
            _ => {
                for (index, op) in program.operators.iter().enumerate() {
                    for (row, col) in present_blocks(op) {
                        out.push(proposal(format!("drop block ({row}, {col}) of {}", op.name), vec![BlockRef { operator: index, row, col }]));
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Coarsen an operator's lattice by `2^j` steps, `j = 0, 1, …`, down to the lattice on which every
/// real rounds to zero: an exponential ladder over the integer precision, so the largest passing
/// coarsening is found in logarithmically many proposals.
pub struct Coarsen;

impl Primitive for Coarsen {
    fn name(&self) -> &'static str {
        "coarsen"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, op) in program.operators.iter().enumerate() {
            let precision = match &op.body {
                OperatorBody::Dense { precision, .. } | OperatorBody::LowRank { precision, .. } => precision,
                OperatorBody::Identity => continue,
            };
            let largest = op.largest_real();
            if largest == 0.0 {
                continue;
            }
            let floor = -(largest.log2().ceil() as i32) - 1;
            let mut step = 1;
            while precision.fraction_bits() - step >= floor {
                let target = DeclaredPrecision::new(precision.fraction_bits() - step).map_err(EngineError::Primitive)?;
                out.push(Proposal {
                    primitive: "coarsen",
                    kind: ProposalKind::Reduce,
                    exactness: Exactness::Approximate,
                    description: format!("coarsen {} from 2^-{} to 2^-{}", op.name, precision.fraction_bits(), target.fraction_bits()),
                    edit: Edit::Precision { operator: index, precision: target },
                });
                step *= 2;
            }
        }
        Ok(out)
    }
}

/// A ReLU unit whose pre-activation is at most zero on every input of the family is the zero law
/// there: its law becomes zero, and the blocks that write it and read it are removed.
pub struct DeadUnits;

impl Primitive for DeadUnits {
    fn name(&self) -> &'static str {
        "dead_units"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, node) in program.nodes.iter().enumerate() {
            let Node::Pointwise { input, laws } = node else { continue };
            let interface = &context.interfaces[*input];
            let pre = &context.trace.values[*input];
            for (group, law) in laws.iter().enumerate() {
                if *law != Law::Relu {
                    continue;
                }
                let range = interface.range(group);
                if pre.slice(s![.., range]).iter().any(|v| *v > 0.0) {
                    continue;
                }
                let mut new_laws = laws.clone();
                new_laws[group] = Law::Zero;
                let mut blocks = Vec::new();
                for (reader, reading) in program.nodes.iter().enumerate() {
                    if let Node::Affine { terms, .. } = reading {
                        for (argument, op) in terms {
                            if *argument == index {
                                for (r, c) in present_blocks(&program.operators[*op]) {
                                    if c == group {
                                        blocks.push(BlockRef { operator: *op, row: r, col: c });
                                    }
                                }
                            }
                        }
                    }
                    if reader == *input
                        && let Node::Affine { terms, bias } = reading
                    {
                        for op in terms.iter().map(|(_, op)| *op).chain(bias.iter().copied()) {
                            for (r, c) in present_blocks(&program.operators[op]) {
                                if r == group {
                                    blocks.push(BlockRef { operator: op, row: r, col: c });
                                }
                            }
                        }
                    }
                }
                out.push(Proposal {
                    primitive: "dead_units",
                    kind: ProposalKind::Reduce,
                    exactness: Exactness::Exact {
                        derivation: format!("relu(t) = 0 for t <= 0 on every family input, unit {group} of node {index}"),
                    },
                    description: format!("dead unit {group} of node {index}"),
                    edit: Edit::Laws { node: index, laws: new_laws, blocks },
                });
            }
        }
        Ok(out)
    }
}

/// Law substitution: any pointwise law may be replaced by any other law of the alphabet
/// ([`super::operator_program::LAWS`]); the two-part code and the contract decide. Level 0 replaces
/// a whole node's laws with one law; level 1 one group's. A zero law also removes the blocks that
/// write and read its group. No substitution is privileged: a sign-gated SiLU-to-ReLU split, a
/// linearized unit and a dead unit are all instances.
pub struct LawSubstitution;

/// The blocks a zero law on `group` of pointwise node `node` leaves unread or unwritten.
fn zero_law_blocks(program: &OperatorProgram, node: usize, input: usize, group: usize) -> Vec<BlockRef> {
    let mut blocks = Vec::new();
    for (reader, reading) in program.nodes.iter().enumerate() {
        let Node::Affine { terms, bias } = reading else { continue };
        for (argument, op) in terms {
            if *argument == node {
                blocks.extend(present_blocks(&program.operators[*op]).into_iter().filter(|(_, c)| *c == group).map(|(row, col)| BlockRef { operator: *op, row, col }));
            }
        }
        if reader == input {
            for op in terms.iter().map(|(_, op)| *op).chain(bias.iter().copied()) {
                blocks.extend(present_blocks(&program.operators[op]).into_iter().filter(|(r, _)| *r == group).map(|(row, col)| BlockRef { operator: op, row, col }));
            }
        }
    }
    blocks
}

impl Primitive for LawSubstitution {
    fn name(&self) -> &'static str {
        "law_substitution"
    }

    fn levels(&self) -> usize {
        2
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, node) in program.nodes.iter().enumerate() {
            let Node::Pointwise { input, laws } = node else { continue };
            let proposal = |description: String, new_laws: Vec<Law>, blocks: Vec<BlockRef>| Proposal {
                primitive: "law_substitution",
                kind: ProposalKind::Reduce,
                exactness: Exactness::Approximate,
                description,
                edit: Edit::Laws { node: index, laws: new_laws, blocks },
            };
            for law in super::operator_program::LAWS {
                if context.level == 0 {
                    if laws.iter().all(|l| *l == law) {
                        continue;
                    }
                    let blocks = if law == Law::Zero {
                        (0..laws.len()).flat_map(|g| zero_law_blocks(program, index, *input, g)).collect()
                    } else {
                        Vec::new()
                    };
                    out.push(proposal(format!("every law of node {index} to {law:?}"), vec![law; laws.len()], blocks));
                } else {
                    for (group, current) in laws.iter().enumerate() {
                        if *current == law {
                            continue;
                        }
                        let mut new_laws = laws.clone();
                        new_laws[group] = law;
                        let blocks = if law == Law::Zero { zero_law_blocks(program, index, *input, group) } else { Vec::new() };
                        out.push(proposal(format!("law of group {group} of node {index} to {law:?}"), new_laws, blocks));
                    }
                }
            }
        }
        Ok(out)
    }
}
