//! The structural cost and the local disagreement of an explanation (#2951).
//!
//! With `P` an explanation of the native model `M` (an executable operator program with rules,
//! [`Artifact`]):
//!
//! * `C(P)` ([`StructuralCost`]) is `P`'s complete structural description in bits: every
//!   independently specified numerical literal at a fixed [`LITERAL_BITS`] (an operator is listed
//!   once however many nodes read it, and a rule body once however many calls apply it), plus the
//!   structure of the program's message (bases, interfaces, present blocks, real counts, rules,
//!   wiring, laws, conditional structure), plus the blocks, places and exceptions that tie it to
//!   `M` ([`Artifact::binding_bits`]). A native computation `P` keeps is charged at its literals
//!   like any other.
//! * `D_local(P)` ([`Local`]) is the native local disagreement: each replaced block run on the
//!   native parent state (`M`'s own values at its reads), its write compared with `M`'s write in the
//!   native interface, `‖write_P(x) − write_M(x)‖₂ / s_b` with `s_b` the block's declared scale (the
//!   root mean square over the declared family of the native write's row norm), worst over the
//!   blocks and the tested inputs. The tested inputs are the declared family and, when declared,
//!   the endpoints of a counterexample ascent ([`Ascent`]) from every input of the family. The
//!   joint write is compared, so two writers' errors that cancel cost nothing and two that align
//!   add.

use super::artifact::{Artifact, inlined};
use super::operator_program::{Basis, FamilyInputs, Node, Operator, OperatorProgram, SequenceLayout, SlotValues};
use ndarray::{Array1, Array2, s};
use std::collections::{BTreeMap, BTreeSet, HashMap};
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
    /// Outward comparison error including the native RMS denominator enclosure.
    /// Covers the final subtraction and norm, not either neural forward computation.
    pub numerical_error: f64,
    /// Envelope of the true per-row maximum, with comparison rounding included.
    pub lower: f64,
    pub upper: f64,
    /// Reported binary64 center of the native RMS scale; its uncertainty is included above.
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
    scales: Mutex<BTreeMap<usize, [f64;3]>>,
    device: Option<(gam_gpu::tensor::Device, usize)>,
    native_device: Option<super::artifact_device::Resident>,
    resident_norms: bool,
}

/// Bounds concern the supplied binary64 writes and their final rounded subtraction,
/// under IEEE RN with gradual underflow; they do not enclose neural-forward error.
/// Max rescaling keeps at least one normalized squared term equal to one.
fn compared_norm(bounds: [f64; 3]) -> Result<(f64, f64), String> {
    let [value, lo, hi] = bounds;
    if hi == 0.0 { return Ok((0.0, 0.0)); }
    // Subtraction of finite binary64 operands has relative error <= u unless
    // exact; subnormal subtraction is exact. Use epsilon >= u conservatively.
    let lower = (lo / (1.0 + f64::EPSILON)).next_down().max(0.0);
    let upper = (hi / (1.0 - f64::EPSILON)).next_up();
    let error = (value - lower).max(upper - value).next_up();
    if !value.is_finite() || !error.is_finite() { return Err("Local norm enclosure unbounded: unresolved".into()); }
    Ok((value, error))
}

fn row_norms(values: &Array2<f64>, columns: std::ops::Range<usize>, scale: [f64;3]) -> Result<Vec<(f64, f64)>, String> {
    values.outer_iter().map(|row| {
        let entries: Vec<_> = row.slice(s![columns.clone()]).iter().copied().collect();
        compared_norm(gam_gpu::tensor::Device::row_l2_enclosure(&entries, scale).map_err(|e|e.to_string())?)
    }).collect()
}

#[derive(Default)]
struct ScaleAccumulator {
    maximum: f64,
    sum: f64,
    lower: f64,
    upper: f64,
}
impl ScaleAccumulator {
    fn add(&mut self, value: f64) -> Result<(), String> {
        if !value.is_finite() { return Err("nonfinite native scale input: unresolved".into()); }
        let value = value.abs();
        if value == 0.0 { return Ok(()); }
        if value > self.maximum {
            if self.maximum == 0.0 {
                self.sum = 0.0; self.lower = 0.0; self.upper = 0.0;
            } else {
                let q = self.maximum / value;
                let qlo = q.next_down().max(0.0); let qhi = q.next_up();
                self.sum *= q * q;
                self.lower = (self.lower * (qlo*qlo).next_down().max(0.0)).next_down().max(0.0);
                self.upper = (self.upper * (qhi*qhi).next_up()).next_up();
            }
            self.maximum = value;
        }
        let q = value / self.maximum;
        let qlo = q.next_down().max(0.0); let qhi = q.next_up();
        self.sum += q*q;
        self.lower = (self.lower + (qlo*qlo).next_down().max(0.0)).next_down().max(0.0);
        self.upper = (self.upper + (qhi*qhi).next_up()).next_up();
        Ok(())
    }
    fn rms(&self, rows: usize) -> Result<[f64;3], String> {
        if rows == 0 || rows > (1_u64 << 53) as usize { return Err("native scale row count is empty or not exact binary64: unresolved".into()); }
        if self.maximum == 0.0 { return Ok([0.0;3]); }
        let rows = rows as f64;
        let center = self.maximum * (self.sum / rows).sqrt();
        let lower = (self.maximum * (self.lower/rows).next_down().max(0.0).sqrt().next_down().max(0.0)).next_down().max(0.0);
        let upper = (self.maximum * (self.upper/rows).next_up().sqrt().next_up()).next_up();
        if !center.is_finite() || !lower.is_finite() || !upper.is_finite() || lower > center || center > upper {
            return Err("native RMS scale enclosure unbounded: unresolved".into());
        }
        Ok([center,lower,upper])
    }
}

impl<'a> Local<'a> {
    pub fn new(model: &'a OperatorProgram, family: FamilyInputs, ascent: Option<Ascent>, batch_rows: usize) -> Self {
        Self { model, family, ascent, batch_rows, scales: Mutex::new(BTreeMap::new()), device: None, native_device: None, resident_norms: false }
    }

    fn scale_enclosure(&self, node: usize) -> Result<[f64;3], String> {
        if let Some(scale) = self.scales.lock().map_err(|e|e.to_string())?.get(&node) { return Ok(*scale); }
        let mut program = self.model.clone();
        program.output = node;
        program.prune();
        let mut accumulator = ScaleAccumulator::default();
        for rows in batches(&units(&self.family), self.batch_rows) {
            let trace = program.execute(&self.family.select(&rows), false).map_err(|e|e.to_string())?;
            for value in &trace.values[program.output] { accumulator.add(*value)?; }
        }
        let scale = accumulator.rms(self.family.rows)?;
        self.scales.lock().map_err(|e|e.to_string())?.insert(node, scale);
        Ok(scale)
    }

    fn block_scales(&self, artifact: &Artifact) -> Result<Vec<[f64;3]>, String> {
        artifact.blocks.iter().map(|b| self.scale_enclosure(b.native_write)).collect()
    }

    /// Per block, per row of `family`, the error and its rounding, from the grafted program.
    fn errors(
        &self,
        local: &Artifact,
        columns: &[std::ops::Range<usize>],
        scales: &[[f64;3]],
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
                        device.scaled_row_l2_enclosed(output, columns.clone(), *scale).map_err(|e|e.to_string())?
                            .into_iter().map(compared_norm).collect::<Result<Vec<_>,String>>()
                    }).collect::<Result<Vec<_>, String>>()?
                } else {
                    let values = device.download(output).map_err(|e| e.to_string())?;
                    columns.iter().zip(scales).map(|(columns, scale)| row_norms(&values, columns.clone(), *scale)).collect::<Result<Vec<_>,String>>()?
                }
            } else {
                let trace = local.execute(&selected)?;
                let values = &trace.values[local.program.output];
                columns.iter().zip(scales).map(|(columns, scale)| row_norms(values, columns.clone(), *scale)).collect::<Result<Vec<_>,String>>()?
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
            .map(|((binding, rows), scale)| BlockError::from_rows(binding.name.clone(), &rows, scale[0]))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(LocalMeasure { blocks, rows: family.rows, family_rows: declared, counterexamples })
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
        scales: &[[f64;3]],
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
                    seed[[*row, c]] = output[[*row, c]] / (norm * scales[*block][0]);
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

// -------------------------------------------------------------------------------- acceptance

// -------------------------------------------------------------------------------- the search

