//! One acceptance path for explanations (#2951): the explanation the search accepts is the
//! artifact that is serialized, decoded, executed, intervened on and reported
//! ([`super::artifact`]).
//!
//! # The objective
//!
//! With `P` an explanation of the native model `M` (an executable operator program with rules,
//! [`Artifact`]), the search solves
//!
//! ```text
//! minimise C(P)   subject to   D_local(P) ≤ δ   and   D_run(P) ≤ ε,
//! ```
//!
//! and reports the solutions as a frontier over declared tolerances `(δ, ε)` ([`frontier`]).
//!
//! * `C(P)` ([`StructuralCost`]) is `P`'s complete structural description in bits: every
//!   independently specified numerical literal at a fixed [`LITERAL_BITS`] (an operator is listed
//!   once however many nodes read it, and a rule body once however many calls apply it), plus the
//!   structure of the program's message (`CodeAccount::structure_bits`: bases, interfaces, present
//!   blocks, real counts, rules, wiring, laws, conditional structure), plus the blocks, places and
//!   exceptions that tie it to `M` ([`Artifact::binding_bits`]). A native computation `P` keeps is
//!   charged at its literals like any other. No lattice index is charged: rounding a literal changes
//!   no length, so quantizing a program is never a discovery. A number inside a node (a gain
//!   coefficient) is in the structure at its exact code and is charged as a literal again.
//! * `D_local(P)` ([`Local`]) is the native local disagreement: each replaced block run on the
//!   native parent state (`M`'s own values at its reads), its write compared with `M`'s write in the
//!   native interface, `‖write_P(x) − write_M(x)‖₂ / s_b` with `s_b` the block's declared scale (the
//!   root mean square over the declared family of the native write's row norm), worst over the
//!   blocks and the tested inputs. The tested inputs are the declared family and, when declared,
//!   the endpoints of a counterexample ascent ([`Ascent`]) from every input of the family and pool.
//!   The joint write is compared, so two writers' errors that cancel cost nothing and two that align
//!   add; a state that differs from the native one in a direction the final norm and unembedding
//!   cannot see still differs.
//! * `D_run(P)` ([`RunCheck`]) is the composed disagreement: `P` executed autonomously on its own
//!   states, clean and under declared native interventions applied at the places `P` holds,
//!   `KL(M ‖ P)` of the output distributions per token, the worst over the declared episode groups
//!   of the group's mean. A place `P` does not hold runs clean in `P`: it predicts no effect.
//!
//! Both disagreements are measured on the decoded artifact (`precision::decode_then_evaluate`) and
//! carry the evidence [`DecodedFidelity`] reads: a candidate is accepted only when both verdicts are
//! `Meets` and its `C` is smaller than the current explanation's. `Violates` and `Unresolved` are
//! both refusals. The per-input record of which groups ran (the activity listing) is determined by
//! `P` and the input, so it is not part of `C`: [`ExecutionCost`] reports it separately.
//!
//! The figures are statements about the two binary64 computations as executed: `M`'s and the
//! decoded `P`'s. Each status's numerical error is the rounding of the comparison itself (the
//! difference, the norm, the log-softmaxes and the KL sum), derived from the operations performed.
//!
//! # The search
//!
//! Every [`Proposer`] offers whole candidate artifacts ([`Proposal`]). Each candidate's `C` is
//! computed exactly, and candidates are tried in the order of the saving `C(current) − C(candidate)`,
//! the quantity acceptance compares. A candidate that saves nothing is never tried. Before
//! decoding, a candidate's local disagreement on the declared family alone is measured; the
//! certified `D_local` is over a superset of those inputs, so one already above `δ` is refused
//! without decoding. The first candidate whose decoded artifact meets both tolerances is accepted,
//! and the search proposes again from it. A block's `D_local` reads only native parent states, so a
//! proposal refused for it is never tried again; one refused for `D_run` is tried again after the
//! next acceptance (composition changes). The search ends when no candidate is accepted or the
//! declared budget is spent. It starts from the native model as its own explanation, which meets
//! every tolerance, so a start such as a decomposition's coordinates enters as a proposal and can
//! never leave the result worse.
//!
//! # The baseline
//!
//! [`code_plus_kl`] is the superseded score `L(P) + n Σ KL(M ‖ P)/ln 2` (lattice-coded program bits
//! plus the data bits of `n` observations per row, no tolerance). It is kept as a labelled baseline
//! and decides nothing.

use super::artifact::{Artifact, EncodedArtifact, inlined};
use super::operator_program::{Basis, FamilyInputs, Law, Node, Operator, OperatorProgram, SequenceLayout, SlotValues};
use super::precision::{DecodedFidelity, FidelityVerdict, decode_then_evaluate_pair};
use super::supports::{EvidenceStatus, ExactBasis, Extremum};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use ndarray::{Array1, Array2, s};
use statrs::function::gamma::ln_gamma;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f64::consts::LN_2;
use std::sync::{Arc, Mutex};

/// The bits of one independently specified numerical literal.
pub const LITERAL_BITS: u64 = 32;

// ------------------------------------------------------------------------------------------ C(P)

/// `C(P)` (module note): structure, literals and the ties to the native model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StructuralCost {
    /// Independently specified numerical literals.
    pub literals: u64,
    /// Numeric-free program structure: real and arithmetic integer payloads excluded.
    /// This differs from the existing wire `CodeAccount::structure_bits`.
    pub structure_bits: u64,
    /// Blocks, places and exceptions ([`Artifact::binding_bits`]).
    pub binding_bits: u64,
}

impl StructuralCost {
    pub fn literal_bits(&self) -> u64 {
        self.literals * LITERAL_BITS
    }

    /// `C(P)` in bits.
    pub fn total(&self) -> u64 {
        self.structure_bits + self.literal_bits() + self.binding_bits
    }
}

/// Operator prices only. Operators are immutable and held by `Arc`, so these prices can be
/// reused across evaluation contexts. Fidelity measurements are deliberately not cached here:
/// they depend on the native model, Local family/interfaces and complete Run protocol. A finite
/// bank retains each assessment explicitly and reuses its evidence across its tolerance grid.
#[derive(Default)]
pub struct CostCache {
    operators: HashMap<usize, (Arc<Operator>, u64, u64)>,
}

/// `C32(artifact)`: numeric-free structure plus 32 bits per independently
/// transmitted numeric literal. The exact wire codec may take a different number of
/// bits, notably for exact architecture epsilons and integer arithmetic knobs.
/// Dimensions, ranks, indices and primitive opcode selections remain structure.
/// Explicit matrix-rule bodies are transmitted and charged once in binding bits.
/// Legacy Copy/Match tags condition cost on decoder-provided template formulas.
pub fn structural_cost(artifact: &Artifact, cache: &mut CostCache) -> Result<StructuralCost, String> {
    // The program as its message holds it: a derived operator with no reals.
    let program = &artifact.message_program()?;
    let derived: BTreeSet<usize> = artifact.derived.iter().map(|d| d.operator).collect();
    let (header, bases, rules, nodes) = program.frame_bits().map_err(|e| e.to_string())?;
    let (node_literals, node_numeric_payload) = program.frame_literal_payload().map_err(|e| e.to_string())?;
    let mut structure_bits = (header + bases.iter().sum::<u64>() + rules + nodes)
        .checked_sub(node_numeric_payload).ok_or("node numeric payload exceeds program frame")?;
    let mut literals = artifact.derived_literals()? + artifact.exceptions.len() as u64 + node_literals;
    for (index, op) in program.operators.iter().enumerate() {
        let key = Arc::as_ptr(op) as usize;
        let (structure, reals) = match cache.operators.get(&key) {
            Some((_, structure, reals)) => (*structure, *reals),
            None => {
                let structure = op.structure_bits().map_err(|e| e.to_string())?;
                let reals = op.real_count() as u64;
                // A derived operator's blank is made anew for every measure: it is not kept.
                if !derived.contains(&index) {
                    cache.operators.insert(key, (op.clone(), structure, reals));
                }
                (structure, reals)
            }
        };
        structure_bits += structure;
        literals += reals;
    }
    Ok(StructuralCost { literals, structure_bits, binding_bits: artifact.binding_bits()? })
}

// -------------------------------------------------------------------------------- execution cost

/// The activity listing (module note): per input, which gated groups of `P` ran. Reported, never
/// charged.
///
/// Every group of a pointwise node whose law is not the zero law is an instance (a call counted as
/// its body inlined), active on an input when its value there is not identically zero. With `c_r`
/// the inputs instance `r` fires on (of `N`), `R` instances, `T = Σ c_r` and `k_x` the instances
/// active on input `x`, the listing is `R log₂(N + 1) + N log₂(R + 1) + Σ_r c_r log₂(T / c_r) −
/// Σ_x log₂ k_x!` bits.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ExecutionCost {
    pub instances: usize,
    pub inputs: usize,
    pub bits: f64,
    /// Mean instances active per input.
    pub mean_active: f64,
}

impl ExecutionCost {
    pub fn bits_per_input(&self) -> f64 {
        if self.inputs == 0 { 0.0 } else { self.bits / self.inputs as f64 }
    }
}

/// `artifact`'s activity listing on `family`, executed `batch_rows` rows (whole units) at a time.
pub fn execution_cost(artifact: &Artifact, family: &FamilyInputs, batch_rows: usize) -> Result<ExecutionCost, String> {
    let flat = inlined(&artifact.program)?;
    let interfaces = flat.interfaces().map_err(|e| e.to_string())?;
    let groups: Vec<(usize, usize)> = flat
        .nodes
        .iter()
        .filter_map(|node| match node {
            Node::Pointwise { input, laws } => Some((*input, laws)),
            _ => None,
        })
        .flat_map(|(input, laws)| laws.iter().enumerate().filter(|(_, law)| **law != Law::Zero).map(move |(group, _)| (input, group)))
        .collect();
    let mut counts = vec![0usize; groups.len()];
    let mut active = Vec::with_capacity(family.rows);
    for rows in batches(&units(family), batch_rows) {
        let inputs = family.select(&rows);
        let trace = flat.execute(&inputs, false).map_err(|e| e.to_string())?;
        let mut fired = vec![0usize; rows.len()];
        for (instance, &(input, group)) in groups.iter().enumerate() {
            let values = &trace.values[input];
            for (row, fired) in fired.iter_mut().enumerate() {
                if interfaces[input].range(group).any(|column| values[[row, column]] != 0.0) {
                    counts[instance] += 1;
                    *fired += 1;
                }
            }
        }
        active.extend(fired);
    }
    let (n, r) = (family.rows as f64, groups.len() as f64);
    let t = counts.iter().sum::<usize>() as f64;
    let mut bits = r * (n + 1.0).log2() + n * (r + 1.0).log2();
    for &c in &counts {
        if c > 0 {
            bits += c as f64 * (t / c as f64).log2();
        }
    }
    for &k in &active {
        bits -= ln_gamma(k as f64 + 1.0) / LN_2;
    }
    let mean_active = if active.is_empty() { 0.0 } else { active.iter().sum::<usize>() as f64 / active.len() as f64 };
    Ok(ExecutionCost { instances: groups.len(), inputs: family.rows, bits, mean_active })
}

// -------------------------------------------------------------------------------- families

/// A family's units: a per-position family's sequences (rows ascending in position), else each row
/// alone.
pub fn units(family: &FamilyInputs) -> Vec<Vec<usize>> {
    match &family.layout {
        None => (0..family.rows).map(|row| vec![row]).collect(),
        Some(layout) => {
            let mut by_sequence: BTreeMap<u32, Vec<(u32, usize)>> = BTreeMap::new();
            for row in 0..family.rows {
                by_sequence.entry(layout.sequence[row]).or_default().push((layout.position[row], row));
            }
            by_sequence
                .into_values()
                .map(|mut rows| {
                    rows.sort_unstable();
                    rows.into_iter().map(|(_, row)| row).collect()
                })
                .collect()
        }
    }
}

/// Whole units in batches of at most `rows` rows (a longer unit alone).
fn batches(units: &[Vec<usize>], rows: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for unit in units {
        if !current.is_empty() && current.len() + unit.len() > rows.max(1) {
            out.push(std::mem::take(&mut current));
        }
        current.extend(unit);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// One slot's value at one row.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotValue {
    Token(u32),
    Raw(Array1<f64>),
}

/// One unit of input: its rows (a sequence's positions in order, or one row), each a value per
/// slot.
pub type Unit = Vec<Vec<SlotValue>>;

/// Unit `rows` of `family` (rows in the order given).
pub fn unit_of(family: &FamilyInputs, rows: &[usize]) -> Unit {
    rows.iter()
        .map(|&row| {
            family
                .slots
                .iter()
                .map(|slot| match slot {
                    SlotValues::Tokens(tokens) => SlotValue::Token(tokens[row]),
                    SlotValues::Raw(values) => SlotValue::Raw(values.row(row).to_owned()),
                })
                .collect()
        })
        .collect()
}

/// The family of `units` in order; with `layout`, unit `i` is sequence `i` with its rows at
/// positions `0, 1, …`.
pub fn family_of(units: &[Unit], layout: bool) -> Result<FamilyInputs, String> {
    let first = units.first().and_then(|u| u.first()).ok_or("an empty family")?;
    let rows: Vec<&Vec<SlotValue>> = units.iter().flatten().collect();
    let slots = (0..first.len())
        .map(|j| match &first[j] {
            SlotValue::Token(_) => rows
                .iter()
                .map(|row| match &row[j] {
                    SlotValue::Token(t) => Ok(*t),
                    SlotValue::Raw(_) => Err(format!("slot {j} mixes tokens and raw rows")),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(SlotValues::Tokens),
            SlotValue::Raw(x) => {
                let mut values = Array2::<f64>::zeros((rows.len(), x.len()));
                for (r, row) in rows.iter().enumerate() {
                    match &row[j] {
                        SlotValue::Raw(v) if v.len() == x.len() => values.row_mut(r).assign(v),
                        _ => return Err(format!("slot {j} mixes widths or kinds")),
                    }
                }
                Ok(SlotValues::Raw(values))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let layout = layout.then(|| {
        let mut sequence = Vec::new();
        let mut position = Vec::new();
        for (i, unit) in units.iter().enumerate() {
            for p in 0..unit.len() {
                sequence.push(i as u32);
                position.push(p as u32);
            }
        }
        SequenceLayout { sequence, position }
    });
    Ok(FamilyInputs { rows: rows.len(), slots, layout })
}

// -------------------------------------------------------------------------------- D_local

/// What one slot of one row may hold.
#[derive(Clone, Debug, PartialEq)]
pub enum SlotDomain {
    /// Any of these tokens.
    Tokens(Vec<u32>),
    /// Any vector in the closed box `[lower, upper]`.
    Box { lower: Array1<f64>, upper: Array1<f64> },
}

/// The counterexample search for `D_local`: the declared input domain (one [`SlotDomain`] per slot,
/// the same at every row), extra starting units, and the exact evaluations each ascent step may
/// spend.
///
/// From each start, a step relaxes tokens to the simplex over their allowed set and raw slots to
/// their box, differentiates the start's worst block error through the grafted program
/// (`derivatives::vjp`), and evaluates exactly, in one batch, the linear oracle's vertex (every
/// row's best token and every raw slot's box corner along the gradient) and the single-row moves of
/// largest linearised gain, as many as the step's evaluations allow, and the box step toward the
/// oracle at `γ = 2^-k` down to the float resolution. The step moves to the best evaluated input
/// when it raises the start's worst error; an ascent stops when none does. Every endpoint is a real
/// input of the domain, evaluated natively; the relaxation only proposes.
#[derive(Clone, Debug)]
pub struct Ascent {
    pub domain: Vec<SlotDomain>,
    pub pool: Vec<Unit>,
    pub evaluations: usize,
}

/// One block's measured disagreement.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BlockError {
    pub name: String,
    /// Worst `‖write_P − write_M‖₂ / s_b` over the tested rows, and that row.
    pub worst: f64,
    pub row: usize,
    /// Row attaining the largest certified lower bound.
    pub lower_row: usize,
    pub numerical_error: f64,
    /// Envelope of the true per-row maximum, with comparison rounding included.
    pub lower: f64,
    pub upper: f64,
    pub scale: f64,
}

/// What [`Local::measure`] found on a decoded artifact.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct LocalMeasure {
    pub blocks: Vec<BlockError>,
    /// Tested rows: the declared family's and the counterexample ascent's endpoints'.
    pub rows: usize,
    pub family_rows: usize,
    /// Ascent endpoints whose worst error exceeds every declared input's.
    pub counterexamples: usize,
}

fn nonnegative_interval(value: f64, error: f64) -> Result<(f64, f64), String> {
    if !value.is_finite() || !error.is_finite() || error < 0.0 {
        return Err("non-finite value or invalid numerical error".to_string());
    }
    let lower = if error == 0.0 { value } else { (value - error).next_down() };
    let upper = if error == 0.0 { value } else { (value + error).next_up() };
    if upper < 0.0 {
        return Err("comparison interval excludes every nonnegative disagreement".to_string());
    }
    Ok((lower.max(0.0), upper))
}

fn maximum_bounds(values: impl IntoIterator<Item = (f64, f64, String)>, domain: String) -> Result<EvidenceStatus<String, String>, String> {
    let (mut lower, mut upper, mut witness) = (0.0_f64, 0.0_f64, None);
    for (lo, hi, name) in values {
        if !lo.is_finite() || hi.is_nan() || lo < 0.0 || hi < lo {
            return Err(format!("{name}: invalid nonnegative disagreement interval"));
        }
        if lo >= lower {
            lower = lo;
            witness = Some(name);
        }
        upper = upper.max(hi);
    }
    EvidenceStatus::unresolved(lower, upper, Extremum::Supremum, witness, domain).map_err(|e| e.to_string())
}

impl BlockError {
    fn from_rows(name: String, rows: &[(f64, f64)], scale: f64) -> Result<Self, String> {
        let (mut worst, mut numerical_error, mut lower, mut upper, mut worst_row, mut witness_row) = (0.0, 0.0, 0.0_f64, 0.0_f64, 0, 0);
        for (row, &(value, error)) in rows.iter().enumerate() {
            let (lo, hi) = nonnegative_interval(value, error).map_err(|e| format!("{name} row {row}: {e}"))?;
            if value >= worst {
                worst = value;
                numerical_error = error;
                worst_row = row;
            }
            if lo >= lower {
                lower = lo;
                witness_row = row;
            }
            upper = upper.max(hi);
        }
        Ok(Self { name, worst, row: worst_row, lower_row: witness_row, numerical_error, lower, upper, scale })
    }
}

impl LocalMeasure {
    /// The worst block and its error with its numerical error.
    pub fn worst(&self) -> Option<&BlockError> {
        self.blocks.iter().max_by(|a, b| a.worst.total_cmp(&b.worst))
    }

    /// Validate every recorded interval and aggregate the worst bound over tested native-parent states.
    /// This describes tested rows only, including any recorded ascent endpoints; it is not a
    /// uniform certificate over untested inputs. Invalid evidence is an error, never a verdict.
    pub fn status(&self) -> Result<EvidenceStatus<String, String>, String> {
        let domain = format!("native parent states of {} tested rows ({} declared)", self.rows, self.family_rows);
        if self.blocks.is_empty() {
            return EvidenceStatus::exact(0.0, 0.0, ExactBasis::Algebraic, None, "no block replaced".to_string()).map_err(|e| e.to_string());
        }
        for block in &self.blocks {
            nonnegative_interval(block.worst, block.numerical_error)?;
        }
        maximum_bounds(
            self.blocks.iter().map(|block| (block.lower, block.upper, format!("{} at tested row {}", block.name, block.lower_row))),
            domain,
        )
    }
}

/// The native side of `D_local`: the model, the declared family, each native write's declared
/// scale (computed once on the family), and the counterexample search. Each row's
/// Euclidean write error is divided by the RMS native-write row norm over the
/// whole declared family, rather than by that particular row's native norm.
pub struct Local<'a> {
    pub model: &'a OperatorProgram,
    pub family: FamilyInputs,
    pub ascent: Option<Ascent>,
    /// Rows executed at once (whole units).
    pub batch_rows: usize,
    scales: Mutex<BTreeMap<usize, f64>>,
    device: Option<(gam_gpu::tensor::Device, usize)>,
    native_device: Option<super::artifact_device::Resident>,
    resident_norms: bool,
}

/// Per row, `‖d_row‖₂` and the rounding of computing it from the two executed writes: the
/// difference rounds once per entry, the sum of `w` squares within `γ_w`, the root and the scaling
/// once each, so the computed norm is within `γ_{w+4}` of the norm of the exact difference.
fn row_norms(values: &Array2<f64>, columns: std::ops::Range<usize>, scale: f64) -> Vec<(f64, f64)> {
    let growth = accumulation_growth(columns.len() + 4);
    values
        .outer_iter()
        .map(|row| {
            let norm = row.slice(s![columns.clone()]).iter().map(|v| v * v).sum::<f64>().sqrt() / scale;
            (norm, (growth * norm).next_up())
        })
        .collect()
}

impl<'a> Local<'a> {
    pub fn new(model: &'a OperatorProgram, family: FamilyInputs, ascent: Option<Ascent>, batch_rows: usize) -> Self {
        Self { model, family, ascent, batch_rows, scales: Mutex::new(BTreeMap::new()), device: None, native_device: None, resident_norms: false }
    }

    /// Execute the same native-parent graft on a float64 CUDA device. Native scale
    /// measurements and comparison reductions stay on the CPU. The limit covers
    /// retained intermediate values, not operators, workspaces or host storage.
    /// There is no CPU fallback. Report this backend: comparison-rounding intervals
    /// describe the executed CUDA values, not a proof of CPU/CUDA equivalence.
    /// Counterexample ascent is currently CPU-only and must be disabled explicitly.
    pub fn with_cuda(mut self, device: gam_gpu::tensor::Device, intermediate_bytes_limit: usize) -> Result<Self, String> {
        if !cfg!(target_os = "linux") || device.is_host() || !device.float64() || intermediate_bytes_limit == 0 {
            return Err("Local CUDA requires Linux float64 accelerator and a positive intermediate byte limit".into());
        }
        if self.ascent.is_some() {
            return Err("Local CUDA does not support counterexample ascent".into());
        }
        if self.native_device.is_some() { return Err("cannot replace Local CUDA device after native source initialization".into()); }
        self.device = Some((device, intermediate_bytes_limit));
        Ok(self)
    }

    /// Optional immutable native operator resident. It uses `self.model` directly,
    /// preserving native f64 values; no precision projection or encode/decode occurs.
    /// Grafts already retain native operator Arcs across pruning and index changes.
    /// Only one source is retained, never all candidate graphs or their traces.
    /// The limit covers float64 operator buffers only, excluding host weights,
    /// indices, activations, workspaces, exception tensors and allocator overhead.
    pub fn with_cuda_native_sharing(mut self, source_numeric_bytes_limit: usize) -> Result<Self, String> {
        if source_numeric_bytes_limit == 0 { return Err("positive native source numeric byte limit required".into()); }
        let (device, _) = self.device.as_ref().ok_or("enable Local CUDA before native sharing")?;
        if self.native_device.is_some() { return Err("Local native source already initialized".into()); }
        let native = Artifact::native(self.model)?;
        self.native_device = Some(super::artifact_device::Resident::from_decoded_values_bounded(device, &native, source_numeric_bytes_limit)?);
        Ok(self)
    }

    pub fn cuda_native_source_numeric_bytes(&self) -> Result<Option<usize>, String> {
        self.native_device.as_ref().map(super::artifact_device::Resident::operator_numeric_bytes).transpose()
    }

    /// Keep the graft's differences on CUDA and download one norm per block/row.
    /// Uses the same ordered binary64 square/sum/sqrt/division and comparison
    /// envelope as the host reduction. Native reference scales remain unchanged.
    pub fn with_cuda_resident_norms(mut self) -> Result<Self, String> {
        if self.device.is_none() { return Err("enable Local CUDA before resident norms".into()); }
        self.resident_norms = true;
        Ok(self)
    }

    pub fn backend_name(&self) -> &'static str {
        if self.resident_norms { return "CUDA f64 native-parent graft and resident row norms; frozen CPU native scales"; }
        if self.native_device.is_some() { "CUDA f64 shared native-parent graft; CPU native scales and comparison" } else if self.device.is_some() { "CUDA f64 native-parent graft; CPU native scales and comparison" } else { "CPU f64 native-parent graft, native scales and comparison" }
    }

    /// The declared scale of native node `node`: the root mean square of its row norms on the
    /// declared family.
    pub fn scale(&self, node: usize) -> Result<f64, String> {
        if let Some(scale) = self.scales.lock().map_err(|e| e.to_string())?.get(&node) {
            return Ok(*scale);
        }
        let mut program = self.model.clone();
        program.output = node;
        program.prune();
        let mut total = 0.0;
        for rows in batches(&units(&self.family), self.batch_rows) {
            let trace = program.execute(&self.family.select(&rows), false).map_err(|e| e.to_string())?;
            total += trace.values[program.output].iter().map(|v| v * v).sum::<f64>();
        }
        let scale = (total / self.family.rows.max(1) as f64).sqrt();
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(format!("native node {node} is zero on the declared family: no scale"));
        }
        self.scales.lock().map_err(|e| e.to_string())?.insert(node, scale);
        Ok(scale)
    }

    fn block_scales(&self, artifact: &Artifact) -> Result<Vec<f64>, String> {
        artifact.blocks.iter().map(|b| self.scale(b.native_write)).collect()
    }

    /// Per block, per row of `family`, the error and its rounding, from the grafted program.
    fn errors(
        &self,
        local: &Artifact,
        columns: &[std::ops::Range<usize>],
        scales: &[f64],
        family: &FamilyInputs,
    ) -> Result<Vec<Vec<(f64, f64)>>, String> {
        let mut out = vec![vec![(0.0, 0.0); family.rows]; columns.len()];
        let resident = match (&self.device, &self.native_device) {
            (Some(_), Some(source)) => Some(super::artifact_device::Resident::from_decoded_values_sharing(source, local)?),
            (Some((device, _)), None) => Some(super::artifact_device::Resident::from_decoded_values(device, local)?),
            (None, _) => None,
        };
        for rows in batches(&units(family), self.batch_rows) {
            let selected = family.select(&rows);
            let errors = if let (Some(resident), Some((device, limit))) = (&resident, &self.device) {
                let estimate = resident.estimated_resident_bytes(selected.rows)?;
                if estimate > *limit {
                    return Err(format!("Local CUDA retained intermediates {estimate} exceed declared limit {limit}; excludes operators/workspaces/host"));
                }
                let trace = resident.forward_edited(&selected, |_, _| Ok(None))?;
                let output = resident.output_ref(&trace)?;
                if self.resident_norms {
                    columns.iter().zip(scales).map(|(columns, scale)| {
                        let growth = accumulation_growth(columns.len() + 4);
                        Ok(device.scaled_row_l2(output, columns.clone(), *scale).map_err(|e| e.to_string())?
                            .into_iter().map(|norm| (norm, (growth * norm).next_up())).collect::<Vec<_>>())
                    }).collect::<Result<Vec<_>, String>>()?
                } else {
                    let values = device.download(output).map_err(|e| e.to_string())?;
                    columns.iter().zip(scales).map(|(columns, scale)| row_norms(&values, columns.clone(), *scale)).collect()
                }
            } else {
                let trace = local.execute(&selected)?;
                let values = &trace.values[local.program.output];
                columns.iter().zip(scales).map(|(columns, scale)| row_norms(values, columns.clone(), *scale)).collect()
            };
            for (b, errors) in errors.into_iter().enumerate() {
                for (row, error) in rows.iter().zip(errors) {
                    nonnegative_interval(error.0, error.1).map_err(|e| format!("local block {b}, row {row}: {e}"))?;
                    out[b][*row] = error;
                }
            }
        }
        Ok(out)
    }

    fn measure_on(&self, artifact: &Artifact, family: &FamilyInputs, declared: usize, counterexamples: usize) -> Result<LocalMeasure, String> {
        artifact.validate_coverage(self.model)?;
        if artifact.blocks.is_empty() {
            return Ok(LocalMeasure { blocks: Vec::new(), rows: family.rows, family_rows: declared, counterexamples });
        }
        let (local, columns) = artifact.local_artifact(self.model)?;
        let scales = self.block_scales(artifact)?;
        let errors = self.errors(&local, &columns, &scales, family)?;
        let blocks = artifact
            .blocks
            .iter()
            .zip(errors)
            .zip(&scales)
            .map(|((binding, rows), scale)| BlockError::from_rows(binding.name.clone(), &rows, *scale))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(LocalMeasure { blocks, rows: family.rows, family_rows: declared, counterexamples })
    }

    /// `D_local` of `artifact` on the declared family alone, without decoding it: a screen.
    pub fn screen(&self, artifact: &Artifact) -> Result<LocalMeasure, String> {
        self.measure_on(artifact, &self.family, self.family.rows, 0)
    }

    /// `D_local` of `artifact` (module note): the declared family and, with an ascent declared, the
    /// ascent endpoints.
    pub fn measure(&self, artifact: &Artifact) -> Result<LocalMeasure, String> {
        let (Some(ascent), false) = (&self.ascent, artifact.blocks.is_empty()) else {
            return self.measure_on(artifact, &self.family, self.family.rows, 0);
        };
        let layout = self.family.layout.is_some();
        let starts: Vec<Unit> = units(&self.family).iter().map(|rows| unit_of(&self.family, rows)).chain(ascent.pool.iter().cloned()).collect();
        let endpoints = self.ascend(artifact, ascent, starts)?;
        let declared = self.measure_on(artifact, &self.family, self.family.rows, 0)?;
        let declared_worst = declared.blocks.iter().map(|b| b.upper).fold(0.0_f64, f64::max);
        let mut found = 0;
        let mut tested = self.family.clone();
        let mut seen: BTreeSet<Vec<u64>> = units(&self.family).iter().map(|rows| key(&unit_of(&self.family, rows))).collect();
        let mut extra = Vec::new();
        for (unit, value, error) in endpoints {
            if seen.insert(key(&unit)) {
                found += usize::from(value - error > declared_worst);
                extra.push(unit);
            }
        }
        if !extra.is_empty() {
            tested = tested.append(&family_of(&extra, layout)?).map_err(|e| e.to_string())?;
        }
        self.measure_on(artifact, &tested, self.family.rows, found)
    }

    /// Per unit, the worst block error over its rows: `(value, its rounding, block, row in unit)`.
    fn unit_worst(
        &self,
        local: &OperatorProgram,
        columns: &[std::ops::Range<usize>],
        scales: &[f64],
        family: &FamilyInputs,
    ) -> Result<Vec<(f64, f64, usize, usize)>, String> {
        let local = Artifact::native(local)?;
        let errors = self.errors(&local, columns, scales, family)?;
        Ok(units(family)
            .iter()
            .map(|rows| {
                let mut best = (0.0, 0.0, 0, 0);
                for (b, block) in errors.iter().enumerate() {
                    for (i, &row) in rows.iter().enumerate() {
                        if block[row].0 > best.0 {
                            best = (block[row].0, block[row].1, b, i);
                        }
                    }
                }
                best
            })
            .collect())
    }

    /// The ascent ([`Ascent`]) from every start: each endpoint with its worst error and rounding.
    fn ascend(&self, artifact: &Artifact, ascent: &Ascent, starts: Vec<Unit>) -> Result<Vec<(Unit, f64, f64)>, String> {
        let layout = self.family.layout.is_some();
        let (local, columns) = artifact.local_program(self.model)?;
        let flat = inlined(&local)?;
        let scales = self.block_scales(artifact)?;
        let mut current: Vec<(Unit, (f64, f64, usize, usize))> = Vec::new();
        for chunk in starts.chunks(self.batch_rows.max(1)) {
            let worst = self.unit_worst(&flat, &columns, &scales, &family_of(chunk, layout)?)?;
            current.extend(chunk.iter().cloned().zip(worst));
        }
        let mut active: Vec<usize> = (0..current.len()).filter(|&i| current[i].1.0 > 0.0).collect();
        while !active.is_empty() {
            let mut next_active = Vec::new();
            for &a in &active {
                let (unit, (value, _, block, row)) = &current[a];
                let family = family_of(std::slice::from_ref(unit), layout)?;
                let trace = flat.execute(&family, false).map_err(|e| e.to_string())?;
                let output = &trace.values[flat.output];
                // The seed: d‖d_row‖/s at the worst block's columns of the worst row.
                let mut seed = Array2::<f64>::zeros(output.dim());
                let range = columns[*block].clone();
                let norm = output.slice(s![*row, range.clone()]).iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm == 0.0 {
                    continue;
                }
                for c in range {
                    seed[[*row, c]] = output[[*row, c]] / (norm * scales[*block]);
                }
                let gradients = slot_gradients(&flat, &family, &trace, seed)?;
                let proposals = candidates(&ascent.domain, unit, &gradients, ascent.evaluations)?;
                if proposals.is_empty() {
                    continue;
                }
                let worst = self.unit_worst(&flat, &columns, &scales, &family_of(&proposals, layout)?)?;
                let best = worst.iter().enumerate().max_by(|x, y| x.1.0.total_cmp(&y.1.0)).map(|(i, w)| (i, *w));
                if let Some((i, w)) = best
                    && w.0 - w.1 > *value
                {
                    current[a] = (proposals[i].clone(), w);
                    next_active.push(a);
                }
            }
            active = next_active;
        }
        Ok(current.into_iter().map(|(unit, (value, error, _, _))| (unit, value, error)).collect())
    }
}

/// A total order key of a unit (raw values by their bits).
fn key(unit: &Unit) -> Vec<u64> {
    let mut out = Vec::new();
    for row in unit {
        for value in row {
            match value {
                SlotValue::Token(t) => out.push(u64::from(*t)),
                SlotValue::Raw(x) => out.extend(x.iter().map(|v| v.to_bits())),
            }
        }
    }
    out
}

/// The gradient of `Σ ⟨seed, output⟩` with respect to every slot's relaxed input: for a token slot
/// `(rows × domain size)` along each token's one-hot, for a raw slot `(rows × width)`; `None` for a
/// slot the program does not read. `program` has no rules. A basis is linear in the one-hot, so a
/// feature node's cotangent `G` reaches its slot as `G Φᵀ`.
fn slot_gradients(
    program: &OperatorProgram,
    family: &FamilyInputs,
    trace: &super::operator_program::Trace,
    seed: Array2<f64>,
) -> Result<Vec<Option<Array2<f64>>>, String> {
    let cotangents = super::derivatives::vjp(program, family, trace, seed).map_err(|e| e.to_string())?;
    let mut slots: Vec<Option<Array2<f64>>> = vec![None; program.declarations.slots.len()];
    for (node, cotangent) in program.nodes.iter().zip(cotangents) {
        let Some(cotangent) = cotangent else { continue };
        let (slot, delta) = match node {
            Node::Feature { slot, basis } => {
                let domain = match &program.bases[*basis] {
                    Basis::Indicator { domain } => *domain,
                };
                let classes: Vec<u32> = (0..program.declarations.domains[domain].size as u32).collect();
                let phi = program.bases[*basis].evaluate(&program.declarations, &classes).map_err(|e| e.to_string())?.values;
                (*slot, cotangent.dot(&phi.t()))
            }
            Node::Raw { slot } => (*slot, cotangent),
            _ => continue,
        };
        match &mut slots[slot] {
            Some(existing) => *existing += &delta,
            empty => *empty = Some(delta),
        }
    }
    Ok(slots)
}

/// The candidates of one ascent step at `unit` ([`Ascent`]).
fn candidates(domain: &[SlotDomain], unit: &Unit, gradients: &[Option<Array2<f64>>], evaluations: usize) -> Result<Vec<Unit>, String> {
    // Single-row token moves (gain, row, slot, token), the oracle vertex, and box directions.
    let mut moves: Vec<(f64, usize, usize, u32)> = Vec::new();
    let mut vertex = unit.clone();
    let mut vertex_moves = 0usize;
    let mut box_direction: Vec<(usize, usize, Array1<f64>, Array1<f64>)> = Vec::new();
    for (r, row) in unit.iter().enumerate() {
        for (j, (slot, value)) in domain.iter().zip(row).enumerate() {
            let Some(g) = gradients.get(j).and_then(Option::as_ref) else { continue };
            let g = g.row(r);
            match (slot, value) {
                (SlotDomain::Tokens(tokens), SlotValue::Token(current)) => {
                    let here = g[*current as usize];
                    let best = tokens
                        .iter()
                        .filter(|t| **t != *current)
                        .map(|t| (g[*t as usize] - here, *t))
                        .filter(|(gain, _)| *gain > 0.0)
                        .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
                    if let Some((gain, t)) = best {
                        moves.push((gain, r, j, t));
                        vertex[r][j] = SlotValue::Token(t);
                        vertex_moves += 1;
                    }
                }
                (SlotDomain::Box { lower, upper }, SlotValue::Raw(x)) => {
                    let target = Array1::from_shape_fn(x.len(), |i| {
                        if g[i] > 0.0 {
                            upper[i]
                        } else if g[i] < 0.0 {
                            lower[i]
                        } else {
                            x[i]
                        }
                    });
                    let gain: f64 = g.iter().zip(target.iter().zip(x)).map(|(gi, (t, xi))| gi * (t - xi)).sum();
                    if gain > 0.0 {
                        box_direction.push((r, j, x.clone(), target));
                    }
                }
                (_, value) => return Err(format!("row {r}, slot {j}: {value:?} is not of the slot's declared domain")),
            }
        }
    }
    moves.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut out = Vec::new();
    if vertex_moves > 1 {
        out.push(vertex);
    }
    for (_, r, j, t) in moves.into_iter().take(evaluations.saturating_sub(out.len())) {
        let mut next = unit.clone();
        next[r][j] = SlotValue::Token(t);
        out.push(next);
    }
    if !box_direction.is_empty() {
        let mut gamma = 1.0_f64;
        loop {
            let mut next = unit.clone();
            let mut moved = false;
            for (r, j, x, target) in &box_direction {
                let y = Array1::from_shape_fn(x.len(), |i| x[i] + gamma * (target[i] - x[i]));
                moved |= y.iter().zip(x).any(|(a, b)| a != b);
                next[*r][*j] = SlotValue::Raw(y);
            }
            if !moved {
                break;
            }
            out.push(next);
            gamma *= 0.5;
        }
    }
    Ok(out)
}

// -------------------------------------------------------------------------------- D_run

/// One declared episode's composed disagreement.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct EpisodeScore {
    pub id: String,
    pub group: String,
    /// Mean `KL(M ‖ P)` per token (nats) over the rows the episode scores, and its rounding.
    pub kl: f64,
    pub numerical_error: f64,
    /// Mean `KL(M ‖ M clean)` per token over the same rows: how much there was to predict.
    pub native_effect: f64,
    /// Share of those rows whose most likely token agrees.
    pub top1_agree: f64,
    /// Number of native edited places `P` does not hold (run clean in `P`); one
    /// composite-site action may name several places.
    pub unheld: usize,
}

/// The composed disagreement of a decoded artifact over declared episodes.
pub trait RunCheck: Sync {
    fn episodes(&self, artifact: &Artifact) -> Result<Vec<EpisodeScore>, String>;

    /// Preserve a backend's directed group aggregation when it has stronger
    /// evidence than averaging rounded episode summaries on the host. Overrides
    /// must describe the same declared episodes and groups. Legacy evaluators
    /// retain their existing arithmetic through this default implementation.
    fn measure(&self, artifact: &Artifact) -> Result<RunMeasure, String> {
        self.episodes(artifact).map(RunMeasure::of)
    }
}

/// What [`RunCheck`] found, by group.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct RunMeasure {
    pub episodes: Vec<EpisodeScore>,
    /// Per group: (group, mean KL per token, mean rounding, mean native effect, episodes).
    pub groups: Vec<(String, f64, f64, f64, usize)>,
}

impl RunMeasure {
    pub fn of(episodes: Vec<EpisodeScore>) -> Self {
        let mut by: BTreeMap<String, Vec<&EpisodeScore>> = BTreeMap::new();
        for e in &episodes {
            by.entry(e.group.clone()).or_default().push(e);
        }
        let groups = by
            .into_iter()
            .map(|(group, es)| {
                let n = es.len() as f64;
                let mean = |f: &dyn Fn(&EpisodeScore) -> f64| es.iter().map(|e| f(e)).sum::<f64>() / n;
                (group, mean(&|e| e.kl), mean(&|e| e.numerical_error), mean(&|e| e.native_effect), es.len())
            })
            .collect();
        Self { episodes, groups }
    }

    /// `D_run`: the worst group's mean, with its rounding.
    pub fn worst(&self) -> Option<&(String, f64, f64, f64, usize)> {
        self.groups.iter().max_by(|a, b| a.1.total_cmp(&b.1))
    }

    fn status(&self) -> Result<EvidenceStatus<String, String>, String> {
        if self.episodes.is_empty() || self.groups.is_empty() {
            return Err("no episode declared".to_string());
        }
        for episode in &self.episodes {
            nonnegative_interval(episode.kl, episode.numerical_error).map_err(|e| format!("episode {}: {e}", episode.id))?;
            if !episode.native_effect.is_finite() || !episode.top1_agree.is_finite() {
                return Err(format!("episode {}: non-finite ancillary measurement", episode.id));
            }
        }
        let groups = self
            .groups
            .iter()
            .map(|group| {
                let (lo, hi) = nonnegative_interval(group.1, group.2)?;
                Ok((lo, hi, format!("group {}", group.0)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        maximum_bounds(groups, format!("{} declared episodes in {} groups", self.episodes.len(), self.groups.len()))
    }
}

/// `KL(p ‖ q)` of two logit rows, and a conditional comparison estimate: each log-softmax `z − m − ln Σ e^{z − m}` rounds
/// the shifted exponentials (two ulps each), their sum (`γ_V`), the logarithm (one ulp) and the
/// subtractions, within `η(z) = γ_{V+4}(|m| + |ln Σ| + 1)` per entry; `p = e^{log p}` within
/// `2u + η(z)` relative; the sum `Σ p (log p − log q)` within `γ_{V+2}` of its magnitude.
///
/// These exp/log ULP assumptions are not guaranteed by Rust's `f64::exp`/`ln`
/// contract, which specifies unspecified precision. Relative exponential error
/// also assumes no lost underflow tail. This is the existing operational CPU
/// comparison model, not a proved transcendental enclosure. It does not cover
/// rounding that produced the input logits, output heads, GEMM or neural forwards.
pub fn kl_logits(z: ndarray::ArrayView1<'_, f64>, w: ndarray::ArrayView1<'_, f64>) -> (f64, f64) {
    let classes = z.len();
    let lse = |v: ndarray::ArrayView1<'_, f64>| {
        let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = v.iter().map(|x| (x - m).exp()).sum();
        (m + total.ln(), accumulation_growth(classes + 4) * (m.abs() + total.ln().abs() + 1.0))
    };
    let ((lz, ez), (lw, ew)) = (lse(z), lse(w));
    let growth = accumulation_growth(classes + 2);
    let (mut kl, mut magnitude, mut spread) = (0.0, 0.0, 0.0);
    for (a, b) in z.iter().zip(w.iter()) {
        let (lp, lq) = (a - lz, b - lw);
        let p = lp.exp();
        kl += p * (lp - lq);
        magnitude += p * (lp.abs() + lq.abs());
        spread += p * (2.0 * UNIT_ROUNDOFF + ez) * (lp - lq).abs();
    }
    (kl, (growth * magnitude + ez + ew + spread).next_up())
}

/// A native intervention on one place: `columns` of native node `node` on `rows` (every row when
/// `None`) scaled, or shifted, after the node is computed.
#[derive(Clone, Debug, PartialEq)]
pub struct Edit {
    pub node: usize,
    pub rows: Option<Vec<usize>>,
    pub columns: std::ops::Range<usize>,
    pub change: Change,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Change {
    Scale(f64),
    Add(f64),
}

/// A declared episode: native edits (none for the clean run) and the group it is summarised in.
#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    pub id: String,
    pub group: String,
    pub edits: Vec<Edit>,
}

/// `D_run` on a declared family: both programs run whole on it, each episode's edits applied to
/// `M` at its nodes and to `P` at the places it holds; the output is `readouts` distribution rows
/// per input.
pub struct FamilyRun<'a> {
    pub model: &'a OperatorProgram,
    pub family: FamilyInputs,
    pub readouts: usize,
    pub episodes: Vec<Episode>,
}

fn apply_edits(edits: &[&Edit], value: &mut Array2<f64>) {
    for edit in edits {
        let rows: Vec<usize> = edit.rows.clone().unwrap_or_else(|| (0..value.nrows()).collect());
        for row in rows {
            for column in edit.columns.clone() {
                let v = &mut value[[row, column]];
                *v = match edit.change {
                    Change::Scale(scale) => *v * scale,
                    Change::Add(shift) => *v + shift,
                };
            }
        }
    }
}

impl RunCheck for FamilyRun<'_> {
    fn episodes(&self, artifact: &Artifact) -> Result<Vec<EpisodeScore>, String> {
        crate::native_control::validate(artifact, self.model)?;
        let native = |edits: &[Edit]| -> Result<Array2<f64>, String> {
            let trace = self
                .model
                .execute_edited(&self.family, |node, value, _| {
                    apply_edits(&edits.iter().filter(|e| e.node == node).collect::<Vec<_>>(), value);
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
            Ok(trace.values[self.model.output].clone())
        };
        let clean = native(&[])?;
        let mut out = Vec::new();
        for episode in &self.episodes {
            let reference = native(&episode.edits)?;
            let mapped = crate::native_control::map_edits(artifact, &episode.edits)?;
            let (held, unheld) = (mapped.edits, mapped.unheld);
            let trace = artifact.execute_edited(&self.family, |node, value, _| {
                if let Some(edits) = held.get(&node) {
                    apply_edits(&edits.iter().collect::<Vec<_>>(), value);
                }
                Ok(())
            })?;
            let explained = &trace.values[artifact.program.output];
            let width = reference.ncols();
            if explained.ncols() != width || width % self.readouts != 0 {
                return Err(format!("outputs of width {width} and {} for {} readouts", explained.ncols(), self.readouts));
            }
            let classes = width / self.readouts;
            let (mut kl, mut error, mut effect, mut agree, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for row in 0..reference.nrows() {
                for k in 0..self.readouts {
                    let range = k * classes..(k + 1) * classes;
                    let (z, w, c) = (reference.slice(s![row, range.clone()]), explained.slice(s![row, range.clone()]), clean.slice(s![row, range]));
                    let (value, rounding) = kl_logits(z, w);
                    kl += value;
                    error += rounding;
                    effect += kl_logits(z, c).0;
                    agree += f64::from(u8::from(argmax(z) == argmax(w)));
                    n += 1.0;
                }
            }
            out.push(EpisodeScore {
                id: episode.id.clone(),
                group: episode.group.clone(),
                kl: kl / n,
                numerical_error: (error / n).next_up(),
                native_effect: effect / n,
                top1_agree: agree / n,
                unheld,
            });
        }
        Ok(out)
    }
}

fn argmax(row: ndarray::ArrayView1<'_, f64>) -> usize {
    row.iter().enumerate().fold((0, f64::NEG_INFINITY), |best, (i, v)| if *v > best.1 { (i, *v) } else { best }).0
}

// -------------------------------------------------------------------------------- acceptance

/// The declared tolerances `(δ, ε)`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Constraint {
    pub local: f64,
    pub run: f64,
}

/// Everything one assessment of a candidate shows.
#[derive(Clone, Debug)]
pub struct Assessment {
    pub cost: StructuralCost,
    pub local: DecodedFidelity<String, String>,
    pub local_measure: LocalMeasure,
    pub run: DecodedFidelity<String, String>,
    pub run_measure: RunMeasure,
}

impl Assessment {
    /// Whether both disagreements are proven within their tolerances.
    pub fn meets(&self) -> bool {
        self.local.verdict() == FidelityVerdict::Meets && self.run.verdict() == FidelityVerdict::Meets
    }

    /// `D_local`, `D_run` as measured.
    pub fn disagreements(&self) -> (f64, f64) {
        (self.local_measure.worst().map_or(0.0, |b| b.worst), self.run_measure.worst().map_or(0.0, |g| g.1))
    }
}

/// Assess `artifact` against `constraint`: its `C`, then its encoded message decoded, and both
/// disagreements measured on what the decoder rebuilt. Refused unless every literal is a 32-bit
/// float.
pub fn assess(
    local: &Local<'_>,
    run: &dyn RunCheck,
    artifact: &Artifact,
    constraint: Constraint,
    cache: &mut CostCache,
) -> Result<Assessment, String> {
    assess_once(local, run, artifact, constraint, cache)
}

/// Assess a candidate once, without constructing or hashing a measurement-cache
/// key. Both measures execute the same decoded artifact; retain the returned
/// assessment to reuse its evidence across a declared tolerance grid.
pub fn assess_once(local: &Local<'_>, run: &dyn RunCheck, artifact: &Artifact, constraint: Constraint, cache: &mut CostCache) -> Result<Assessment, String> {
    if !constraint.local.is_finite() || constraint.local < 0.0 || !constraint.run.is_finite() || constraint.run < 0.0 {
        return Err("the fidelity tolerance must be finite and nonnegative".into());
    }
    PreparedAssessment::new(artifact, cache)?.assess(local, run, constraint)
}

/// Optional staged evidence. Local rejection has no Run measurement or verdict.
/// Its exclusion proof applies only at local tolerances no wider than the declared grid.
#[derive(Clone, Debug)]
pub enum StagedAssessment {
    Complete(Assessment),
    LocalRejected {
        cost: StructuralCost,
        local: DecodedFidelity<String, String>,
        local_measure: LocalMeasure,
        max_local_tolerance: f64,
    },
}

impl StagedAssessment {
    pub fn cost(&self) -> StructuralCost {
        match self { Self::Complete(a) => a.cost, Self::LocalRejected { cost, .. } => *cost }
    }

    /// Missing Run is explicit; it is never represented by a zero measure.
    pub fn run_measure(&self) -> Option<&RunMeasure> {
        match self { Self::Complete(a) => Some(&a.run_measure), Self::LocalRejected { .. } => None }
    }

    /// Joint feasibility evidence at a grid point. Outside a rejected stage's
    /// declared maximum local tolerance the assessment remains unresolved.
    pub fn verdict(&self, constraint: Constraint) -> Result<FidelityVerdict, String> {
        validate_constraints(&[constraint])?;
        match self {
            Self::LocalRejected { local, max_local_tolerance, .. } => {
                if constraint.local <= *max_local_tolerance && local.with_tolerance(constraint.local)?.verdict() == FidelityVerdict::Violates {
                    Ok(FidelityVerdict::Violates)
                } else { Ok(FidelityVerdict::Unresolved) }
            }
            Self::Complete(a) => {
                let l = a.local.with_tolerance(constraint.local)?.verdict();
                let r = a.run.with_tolerance(constraint.run)?.verdict();
                Ok(match (l, r) {
                    (FidelityVerdict::Violates, _) | (_, FidelityVerdict::Violates) => FidelityVerdict::Violates,
                    (FidelityVerdict::Meets, FidelityVerdict::Meets) => FidelityVerdict::Meets,
                    _ => FidelityVerdict::Unresolved,
                })
            }
        }
    }
}

fn validate_constraints(constraints: &[Constraint]) -> Result<(), String> {
    if constraints.is_empty() { return Err("a nonempty declared tolerance grid is required".into()); }
    for c in constraints {
        if !c.local.is_finite() || c.local < 0.0 || !c.run.is_finite() || c.run < 0.0 {
            return Err("the fidelity tolerance must be finite and nonnegative".into());
        }
    }
    Ok(())
}

/// Optional exact Local-first screen of each complete candidate message, never
/// of its constituent edits. Decode and check complete cost once, then measure
/// the unchanged full Local family. Only a proved violation at the widest declared
/// delta omits Run. Retain this evidence across grid points without measuring again.
/// The ordinary assessment and finite-bank defaults remain unchanged.
pub fn assess_once_local_first(
    local: &Local<'_>, run: &dyn RunCheck, artifact: &Artifact,
    constraints: &[Constraint], cache: &mut CostCache,
) -> Result<StagedAssessment, String> {
    validate_constraints(constraints)?;
    let prepared = PreparedAssessment::new(artifact, cache)?;
    assess_local_first_decodable(&prepared.encoded, prepared.cost, local, run, constraints)
}

/// The same optional Local-first assessment with fixed-native codec reuse. Its
/// full message, decoded numbers, complete cost and missing Run semantics are unchanged.
pub fn assess_once_local_first_with_native_codec(
    local: &Local<'_>, run: &dyn RunCheck, artifact: &Artifact,
    constraints: &[Constraint], cache: &mut CostCache,
    codec: &super::operator_program::NativeOperatorCodec,
) -> Result<StagedAssessment, String> {
    validate_constraints(constraints)?;
    if !artifact.has_f32_literals() { return Err("a literal that is not a 32-bit float".into()); }
    let cost = structural_cost(artifact, cache)?;
    let encoded = EncodedArtifact::of_with_native_codec(artifact, codec)?;
    assess_local_first_decodable(&encoded.using_native_codec(codec), cost, local, run, constraints)
}

fn assess_local_first_decodable<A: super::precision::DecodableArtifact<Decoded = Artifact>>(
    encoded: &A, cost: StructuralCost, local: &Local<'_>, run: &dyn RunCheck,
    constraints: &[Constraint],
) -> Result<StagedAssessment, String> {
    let max_local = constraints.iter().map(|c| c.local).fold(0.0_f64, f64::max);
    let max_run = constraints.iter().map(|c| c.run).fold(0.0_f64, f64::max);
    let ((local_measure, run_measure), (local_fidelity, run_fidelity)) =
        super::precision::decode_then_evaluate_optional_pair(
            encoded,
            |decoded: &Artifact| {
                if structural_cost(decoded, &mut CostCache::default())? != cost {
                    return Err("decoded artifact has a different structural cost".into());
                }
                let measured = local.measure(decoded)?;
                let rejected = measured.status()?.refutes_at_most(max_local);
                let run_measure = if rejected { None } else { Some(run.measure(decoded)?) };
                Ok((measured, run_measure))
            },
            |(local, run): &(LocalMeasure, Option<RunMeasure>)| {
                Ok((local.status()?, run.as_ref().map(RunMeasure::status).transpose()?))
            },
            [max_local, max_run],
        )?;
    match (run_measure, run_fidelity) {
        (Some(run_measure), Some(run_fidelity)) => Ok(StagedAssessment::Complete(Assessment {
            cost, local: local_fidelity, local_measure, run: run_fidelity, run_measure,
        })),
        (None, None) => Ok(StagedAssessment::LocalRejected {
            cost, local: local_fidelity, local_measure, max_local_tolerance: max_local,
        }),
        _ => Err("inconsistent optional Run evidence".into()),
    }
}

/// The same decoded assessment with bounded, fixed-native operator codec reuse.
/// The message remains standalone; only exactly witnessed native codewords reuse
/// decoding work. This does not cache measurements or change either verdict.
pub fn assess_once_with_native_codec(
    local: &Local<'_>, run: &dyn RunCheck, artifact: &Artifact, constraint: Constraint,
    cache: &mut CostCache, codec: &super::operator_program::NativeOperatorCodec,
) -> Result<Assessment, String> {
    if !constraint.local.is_finite() || constraint.local < 0.0 || !constraint.run.is_finite() || constraint.run < 0.0 {
        return Err("the fidelity tolerance must be finite and nonnegative".into());
    }
    if !artifact.has_f32_literals() {
        return Err("a literal that is not a 32-bit float".into());
    }
    let prepared = PreparedAssessment {
        cost: structural_cost(artifact, cache)?,
        encoded: EncodedArtifact::of_with_native_codec(artifact, codec)?,
    };
    prepared.assess_decodable(&prepared.encoded.using_native_codec(codec), local, run, constraint)
}

/// One immutable message and the structural cost of exactly the artifact that produced it.
/// The finite-bank evaluator can compare this message for deduplication, then decode it for
/// fidelity, without serializing every checkpoint a second time. Private fields prevent pairing
/// an unrelated message with a cheaper cost. At most one such message is retained by the bank.
pub(crate) struct PreparedAssessment {
    encoded: EncodedArtifact,
    cost: StructuralCost,
}

impl PreparedAssessment {
    pub(crate) fn new(artifact: &Artifact, cache: &mut CostCache) -> Result<Self, String> {
        if !artifact.has_f32_literals() {
            return Err("a literal that is not a 32-bit float".into());
        }
        Ok(Self { cost: structural_cost(artifact, cache)?, encoded: EncodedArtifact::of(artifact)? })
    }

    pub(crate) fn message(&self) -> &super::codec::BitString { &self.encoded.message }

    pub(crate) fn assess(&self, local: &Local<'_>, run: &dyn RunCheck, constraint: Constraint) -> Result<Assessment, String> {
        self.assess_decodable(&self.encoded, local, run, constraint)
    }

    fn assess_decodable<A: super::precision::DecodableArtifact<Decoded = Artifact>>(
        &self, encoded: &A, local: &Local<'_>, run: &dyn RunCheck, constraint: Constraint,
    ) -> Result<Assessment, String> {
        let ((local_measure, run_measure), [local, run]) = decode_then_evaluate_pair(
            encoded,
            |decoded: &Artifact| {
                // A fresh cache releases these independent decoded operators after this
                // measurement rather than retaining one full model per bank candidate.
                let decoded_cost = structural_cost(decoded, &mut CostCache::default())?;
                if decoded_cost != self.cost {
                    return Err("decoded artifact has a different structural cost".into());
                }
                Ok((local.measure(decoded)?, run.measure(decoded)?))
            },
            |(local, run)| Ok([local.status()?, run.status()?]),
            [constraint.local, constraint.run],
        )?;
        Ok(Assessment { cost: self.cost, local, local_measure, run, run_measure })
    }
}

// -------------------------------------------------------------------------------- the search

/// A candidate explanation offered to the search.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub source: String,
    /// Stable: a proposal refused for `D_local` is not tried again under this description.
    pub description: String,
    pub candidate: Artifact,
}

/// What a proposer sees.
pub struct Context<'a> {
    pub model: &'a OperatorProgram,
    pub current: &'a Artifact,
    pub cost: StructuralCost,
    pub constraint: Constraint,
}

/// A source of candidate explanations.
pub trait Proposer {
    fn name(&self) -> &str;
    fn propose(&self, context: &Context<'_>) -> Result<Vec<Proposal>, String>;
}

/// The search's declared budget: candidates decoded and certified, and rounds of proposals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Budget {
    pub certifications: usize,
    pub rounds: usize,
}

/// What became of one tried candidate.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub enum Outcome {
    Accepted,
    /// The family alone already shows `D_local` above `δ`.
    LocalScreen(f64),
    LocalViolates(f64),
    LocalUnresolved(f64),
    RunViolates(f64),
    RunUnresolved(f64),
    Refused(String),
}

/// One tried candidate.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Step {
    pub source: String,
    pub description: String,
    pub saving: u64,
    pub cost: u64,
    pub outcome: Outcome,
}

/// Why the search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Stop {
    /// No candidate that saves bits met both tolerances.
    Converged,
    CertificationBudget,
    RoundBudget,
}

/// A search's result: the accepted artifact, its assessment, and every tried candidate.
#[derive(Clone, Debug)]
pub struct Searched {
    pub artifact: Artifact,
    pub assessment: Assessment,
    pub steps: Vec<Step>,
    pub stop: Stop,
}

fn verdict_outcome(assessment: &Assessment) -> Outcome {
    let (d_local, d_run) = assessment.disagreements();
    match (assessment.local.verdict(), assessment.run.verdict()) {
        (FidelityVerdict::Violates, _) => Outcome::LocalViolates(d_local),
        (FidelityVerdict::Unresolved, _) => Outcome::LocalUnresolved(d_local),
        (FidelityVerdict::Meets, FidelityVerdict::Violates) => Outcome::RunViolates(d_run),
        (FidelityVerdict::Meets, FidelityVerdict::Unresolved) => Outcome::RunUnresolved(d_run),
        (FidelityVerdict::Meets, FidelityVerdict::Meets) => Outcome::Accepted,
    }
}

/// The search (module note) from `start`, which must meet `constraint` (the native model as its own
/// explanation always does).
pub fn search(
    local: &Local<'_>,
    run: &dyn RunCheck,
    proposers: &[&dyn Proposer],
    start: &Artifact,
    constraint: Constraint,
    budget: Budget,
) -> Result<Searched, String> {
    search_with(local, run, proposers, start, constraint, budget, &mut CostCache::default())
}

/// [`search`] reusing immutable operator prices from earlier searches (`cache`).
/// Fidelity is measured in the supplied context; evidence from other contexts is never reused.
pub fn search_with(
    local: &Local<'_>,
    run: &dyn RunCheck,
    proposers: &[&dyn Proposer],
    start: &Artifact,
    constraint: Constraint,
    budget: Budget,
    cache: &mut CostCache,
) -> Result<Searched, String> {
    let mut current = start.f32_literals()?;
    let mut assessment = assess(local, run, &current, constraint, cache)?;
    if !assessment.meets() {
        return Err(format!("the start does not meet the tolerances: {:?}", verdict_outcome(&assessment)));
    }
    let mut steps = Vec::new();
    let mut refused_local: BTreeSet<String> = BTreeSet::new();
    let mut refused: BTreeSet<String> = BTreeSet::new();
    let (mut certifications, mut rounds) = (0usize, 0usize);
    let stop = 'search: loop {
        if rounds >= budget.rounds {
            break Stop::RoundBudget;
        }
        rounds += 1;
        let context = Context { model: local.model, current: &current, cost: assessment.cost, constraint };
        let mut ranked: Vec<(u64, Proposal, StructuralCost)> = Vec::new();
        for proposer in proposers {
            for proposal in proposer.propose(&context)? {
                if refused_local.contains(&proposal.description) || refused.contains(&proposal.description) {
                    continue;
                }
                let proposal = Proposal { candidate: proposal.candidate.f32_literals()?, ..proposal };
                let cost = structural_cost(&proposal.candidate, cache)?;
                if cost.total() < assessment.cost.total() {
                    ranked.push((assessment.cost.total() - cost.total(), proposal, cost));
                }
            }
        }
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.description.cmp(&b.1.description)));
        let mut accepted = None;
        for (saving, proposal, cost) in ranked {
            let step = |outcome: Outcome| Step {
                source: proposal.source.clone(),
                description: proposal.description.clone(),
                saving,
                cost: cost.total(),
                outcome,
            };
            let screened = match local.screen(&proposal.candidate) {
                Ok(screened) => screened,
                Err(error) => {
                    steps.push(step(Outcome::Refused(error)));
                    refused_local.insert(proposal.description.clone());
                    continue;
                }
            };
            if let Some(worst) = screened.blocks.iter().max_by(|a, b| a.lower.total_cmp(&b.lower))
                && worst.lower > constraint.local
            {
                steps.push(step(Outcome::LocalScreen(worst.worst)));
                refused_local.insert(proposal.description.clone());
                continue;
            }
            if certifications >= budget.certifications {
                break 'search Stop::CertificationBudget;
            }
            certifications += 1;
            let candidate = match assess(local, run, &proposal.candidate, constraint, cache) {
                Ok(candidate) => candidate,
                Err(error) => {
                    steps.push(step(Outcome::Refused(error)));
                    refused_local.insert(proposal.description.clone());
                    continue;
                }
            };
            let outcome = verdict_outcome(&candidate);
            log::info!("{} {}: C {} -> {} bits: {:?}", proposal.source, proposal.description, assessment.cost.total(), cost.total(), outcome);
            steps.push(step(outcome.clone()));
            match outcome {
                Outcome::Accepted => {
                    accepted = Some((proposal.candidate, candidate));
                    break;
                }
                Outcome::LocalViolates(_) | Outcome::LocalUnresolved(_) => {
                    refused_local.insert(proposal.description.clone());
                }
                _ => {
                    refused.insert(proposal.description.clone());
                }
            }
        }
        match accepted {
            Some((artifact, candidate)) => {
                current = artifact;
                assessment = candidate;
                refused.clear();
            }
            None => break Stop::Converged,
        }
    };
    Ok(Searched { artifact: current, assessment, steps, stop })
}

/// One point of the frontier: the tolerances, and the accepted explanation's `C` and
/// disagreements.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FrontierPoint {
    pub local_tolerance: f64,
    pub run_tolerance: f64,
    pub cost: StructuralCost,
    pub local: f64,
    pub run: f64,
    pub blocks: Vec<String>,
    pub stop: Stop,
}

/// The frontier of `C` over the tolerance grid `locals × runs` (module note). Each tolerance pair
/// is searched from the result of the next tighter pair already searched (which meets the looser
/// tolerances too), the tightest from `start`.
pub fn frontier(
    local: &Local<'_>,
    run: &dyn RunCheck,
    proposers: &[&dyn Proposer],
    start: &Artifact,
    locals: &[f64],
    runs: &[f64],
    budget: Budget,
) -> Result<(Vec<FrontierPoint>, Vec<Searched>), String> {
    let mut locals = locals.to_vec();
    let mut runs = runs.to_vec();
    locals.sort_by(f64::total_cmp);
    runs.sort_by(f64::total_cmp);
    let mut results: BTreeMap<(usize, usize), Searched> = BTreeMap::new();
    let mut cache = CostCache::default();
    for (i, &delta) in locals.iter().enumerate() {
        for (j, &epsilon) in runs.iter().enumerate() {
            let from = [(i.wrapping_sub(1), j), (i, j.wrapping_sub(1))]
                .iter()
                .filter_map(|k| results.get(k))
                .min_by_key(|r| r.assessment.cost.total())
                .map_or_else(|| start.clone(), |r| r.artifact.clone());
            let searched = search_with(local, run, proposers, &from, Constraint { local: delta, run: epsilon }, budget, &mut cache)?;
            results.insert((i, j), searched);
        }
    }
    let mut points = Vec::new();
    let mut searches = Vec::new();
    for ((i, j), searched) in results {
        let (d_local, d_run) = searched.assessment.disagreements();
        points.push(FrontierPoint {
            local_tolerance: locals[i],
            run_tolerance: runs[j],
            cost: searched.assessment.cost,
            local: d_local,
            run: d_run,
            blocks: searched.artifact.blocks.iter().map(|b| b.name.clone()).collect(),
            stop: searched.stop,
        });
        searches.push(searched);
    }
    Ok((points, searches))
}

// -------------------------------------------------------------------------------- the baseline

/// The superseded score (module note, "The baseline"), labelled and deciding nothing: `P`'s
/// lattice-coded message length plus `n Σ_rows KL(M ‖ P) / ln 2` over `family`, `readouts`
/// distribution rows per input.
pub fn code_plus_kl(model: &OperatorProgram, artifact: &Artifact, family: &FamilyInputs, readouts: usize, observations: u64) -> Result<f64, String> {
    let bits = artifact.program.code_bits().map_err(|e| e.to_string())? as f64;
    let reference = model.execute(family, false).map_err(|e| e.to_string())?.values[model.output].clone();
    let explained = artifact.execute(family)?.values[artifact.program.output].clone();
    let classes = reference.ncols() / readouts.max(1);
    let mut kl = 0.0;
    for row in 0..reference.nrows() {
        for k in 0..readouts {
            let range = k * classes..(k + 1) * classes;
            kl += kl_logits(reference.slice(s![row, range.clone()]), explained.slice(s![row, range])).0;
        }
    }
    Ok(bits + observations as f64 * kl / LN_2)
}

#[cfg(test)]
mod maximum_evidence_tests {
    use super::*;
    fn local(blocks: &[(f64, f64)]) -> LocalMeasure {
        LocalMeasure {
            blocks: blocks
                .iter()
                .enumerate()
                .map(|(i, &(value, error))| BlockError::from_rows(format!("b{i}"), &[(value, error)], 1.0).unwrap())
                .collect(),
            rows: 1,
            family_rows: 1,
            counterexamples: 0,
        }
    }
    fn run(groups: &[(f64, f64)]) -> RunMeasure {
        RunMeasure::of(
            groups
                .iter()
                .enumerate()
                .map(|(i, &(kl, numerical_error))| EpisodeScore {
                    id: format!("e{i}"),
                    group: format!("g{i}"),
                    kl,
                    numerical_error,
                    native_effect: 0.0,
                    top1_agree: 1.0,
                    unheld: 0,
                })
                .collect(),
        )
    }
    #[test]
    fn smaller_central_value_can_have_the_largest_upper_bound() {
        for status in [local(&[(1.0, 0.0), (0.9, 0.3)]).status().unwrap(), run(&[(1.0, 0.0), (0.9, 0.3)]).status().unwrap()] {
            assert!(!status.certifies_at_most(1.1));
            assert!(status.certifies_at_most(1.3));
        }
    }
    #[test]
    fn smaller_central_value_can_prove_a_violation() {
        for status in [local(&[(1.0, 1.0), (0.9, 0.0)]).status().unwrap(), run(&[(1.0, 1.0), (0.9, 0.0)]).status().unwrap()] {
            assert!(status.refutes_at_most(0.8));
        }
    }
    #[test]
    fn nonmax_records_cannot_hide_invalid_evidence() {
        let negative_nan = f64::from_bits(0xfff8_0000_0000_0000);
        for bad in [(0.0, f64::INFINITY), (negative_nan, 0.0), (0.0, -1.0)] {
            assert!(BlockError::from_rows("bad".to_string(), &[(1.0, 0.0), bad], 1.0).is_err());
            assert!(run(&[(1.0, 0.0), bad]).status().is_err());
        }
        // A bad episode must be checked before aggregation can mask it.
        let mut mixed = run(&[(1.0, 0.0), (0.0, 0.0)]);
        mixed.episodes[1].numerical_error = f64::INFINITY;
        assert!(mixed.status().is_err());
    }
    #[test]
    fn within_one_block_all_row_intervals_are_preserved() {
        let block = BlockError::from_rows("one".to_string(), &[(1.0, 0.6), (0.9, 0.8), (0.8, 0.0)], 1.0).unwrap();
        assert_eq!(block.worst, 1.0);
        assert_eq!(block.lower_row, 2); // Largest lower comes from the exact smaller row.
        assert_eq!(block.lower, 0.8);
        assert!(block.upper > 1.7);
        let measured = LocalMeasure { blocks: vec![block], rows: 3, family_rows: 3, counterexamples: 0 };
        let status = measured.status().unwrap();
        assert!(!status.certifies_at_most(1.65));
        assert!(status.refutes_at_most(0.75));
    }
    #[test]
    fn negative_kl_must_have_an_interval_intersecting_zero() {
        assert!(run(&[(-1.0, 0.01)]).status().is_err());
        assert!(run(&[(-1e-15, 1e-14)]).status().unwrap().certifies_at_most(1e-12));
    }
}
