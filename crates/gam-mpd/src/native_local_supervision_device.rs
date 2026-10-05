//! Prepared device execution for native same-input local supervision.
use super::{Panel, Report, boundary_plan};
use crate::{
    artifact::Artifact,
    artifact_device::mapped_inlined,
    device_program::DeviceProgram,
    operator_program::{FamilyInputs, Interface, Node, Operator, OperatorProgram, Slot, SlotValues, exact_precision},
};
use gam_gpu::tensor::{Device, Tensor};
use ndarray::Array2;
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// A candidate's incoming states and the closed native equation on those same states.
/// Compile once per candidate/native pair and region, then capture complete sequence families.
/// F64 storage is required; unsupported device primitives are errors, never CPU fallbacks.
pub struct PreparedDeviceCapture {
    candidate: DeviceProgram,
    teacher: DeviceProgram,
    teacher_boundaries: BTreeMap<usize, usize>,
    input_widths: Vec<usize>,
    arguments: Vec<usize>,
    outputs: Vec<usize>,
    output_widths: Vec<usize>,
    native_evaluated_nodes: Vec<usize>,
    native_prefix_nodes_avoided: usize,
    operator_numeric_bytes: usize,
    attention_width: usize,
    rotary_width: usize,
    source_slots: Vec<Slot>,
    largest_rows: Cell<usize>,
    largest_rotary_span: Cell<usize>,
}

fn add(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or("device local supervision size overflow".into())
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or("device local supervision size overflow".into())
}

/// Existing pruning retains original operator Arcs. Inlining materializes call-output
/// identities so clamping one invocation never changes another alias of its return value.
fn pruned_values(
    mut program: OperatorProgram,
    outputs: Vec<usize>,
) -> Result<OperatorProgram, String> {
    program.nodes.push(Node::Concat { parts: outputs });
    program.output = program.nodes.len() - 1;
    program.prune();
    let (mut flat, _) = mapped_inlined(&program)?;
    flat.prune();
    flat.interfaces().map_err(|e| e.to_string())?;
    Ok(flat)
}

impl PreparedDeviceCapture {
    /// `operator_numeric_bytes` bounds the conservative SUM of both programs' retained
    /// numeric operator buffers, counting shared buffers twice. Candidate/native exact Arc
    /// owners are shared where possible; this accounting deliberately does not assume a hit.
    /// It excludes host source coefficients, code/index arrays and allocator overhead.
    pub fn new(
        d: &Device,
        native: &OperatorProgram,
        candidate: &Artifact,
        arguments: &[usize],
        outputs: &[usize],
        operator_numeric_bytes: usize,
    ) -> Result<Self, String> {
        if !d.float64() {
            return Err("device local supervision requires float64 storage".into());
        }
        let plan = boundary_plan(native, candidate, arguments, outputs)?;
        if native.declarations.slots.len() != candidate.program.declarations.slots.len() {
            return Err("device local supervision requires matching family slot counts".into());
        }
        let input_widths = arguments
            .iter()
            .map(|n| plan.nt[*n].width())
            .collect::<Vec<_>>();
        let candidate_flat = pruned_values(
            candidate.program.clone(),
            arguments.iter().map(|n| plan.reads[n]).collect(),
        )?;

        // Typed zero placeholders preserve grouped native interfaces. They occur only in
        // this private teacher slice and are replaced by resident candidate values before
        // any downstream use. Original boundary producers and unrelated nodes are pruned.
        let mut teacher = native.clone();
        let mut placeholders = Vec::new();
        for &n in arguments {
            let interface = plan.nt[n].clone();
            let placeholder = Arc::new(
                Operator::dense(
                    "local boundary placeholder",
                    interface.clone(),
                    Interface::constant(),
                    Array2::zeros((interface.width(), 1)),
                    exact_precision([0.]).map_err(|e| e.to_string())?,
                    Default::default(),
                )
                .map_err(|e| e.to_string())?,
            );
            teacher.nodes[n] = Node::Constant {
                operator: teacher.operators.len(),
            };
            teacher.operators.push(Arc::clone(&placeholder));
            placeholders.push(placeholder);
        }
        let teacher_flat = pruned_values(teacher, outputs.to_vec())?;
        let teacher_boundaries = teacher_flat
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(n, node)| {
                let Node::Constant { operator } = node else {
                    return None;
                };
                placeholders
                    .iter()
                    .position(|p| Arc::ptr_eq(p, &teacher_flat.operators[*operator]))
                    .map(|argument| (n, argument))
            })
            .collect();
        let mut attention_width = 0;
        let mut rotary_width = 0;
        for program in [&candidate_flat, &teacher_flat] {
            let interfaces = program.interfaces().map_err(|e| e.to_string())?;
            for node in &program.nodes {
                if let Node::Attend { query, rotary, .. } = node {
                    attention_width = attention_width.max(interfaces[*query].width());
                    if rotary.is_some() {
                        rotary_width = add(rotary_width, interfaces[*query].width())?;
                    }
                }
            }
        }
        let source_slots = candidate.program.declarations.slots.clone();
        let candidate =
            DeviceProgram::compile_values_bounded(d, &candidate_flat, operator_numeric_bytes)?;
        let candidate_bytes = candidate.operator_numeric_bytes()?;
        let remaining = operator_numeric_bytes
            .checked_sub(candidate_bytes)
            .ok_or("device local supervision operator budget exhausted")?;
        let teacher = DeviceProgram::compile_values_sharing_bounded(
            &candidate,
            &teacher_flat,
            remaining.max(1),
        )?;
        let retained = add(candidate_bytes, teacher.operator_numeric_bytes()?)?;
        if retained > operator_numeric_bytes {
            return Err("device local supervision operator budget exhausted".into());
        }
        Ok(Self {
            candidate,
            teacher,
            teacher_boundaries,
            input_widths,
            arguments: arguments.to_vec(),
            outputs: outputs.to_vec(),
            output_widths: plan.output_widths,
            native_prefix_nodes_avoided: plan.last_native + 1 - plan.native_evaluated_nodes.len(),
            native_evaluated_nodes: plan.native_evaluated_nodes,
            operator_numeric_bytes: retained,
            attention_width,
            rotary_width,
            source_slots,
            largest_rows: Cell::new(0),
            largest_rotary_span: Cell::new(0),
        })
    }

    /// Conservative retained numeric operator bytes; shared buffers count in both programs.
    pub fn operator_numeric_bytes(&self) -> usize {
        self.operator_numeric_bytes
    }

    /// Trace/panel plan includes simultaneous host return panels, resident boundary copies,
    /// five trace-sized buffers, conservative attention workspace and rotary tables/gathers.
    /// It excludes separately budgeted operators, caller-owned inputs/retained panels,
    /// source coefficients, code/index arrays and backend/allocator scratch.
    pub fn planned_trace_and_panel_bytes(&self, family: &FamilyInputs) -> Result<usize, String> {
        if family.rows == 0 {
            return Err("local supervision needs nonempty boundaries and rows".into());
        }
        let rows = family.rows.max(self.largest_rows.get());
        let trace = mul(
            add(self.candidate.bytes_per_row(), self.teacher.bytes_per_row())?,
            rows,
        )?;
        let input_width = self
            .input_widths
            .iter()
            .try_fold(0, |sum, w| add(sum, *w))?;
        let output_width = self
            .output_widths
            .iter()
            .try_fold(0, |sum, w| add(sum, *w))?;
        let panels = mul(
            mul(
                add(mul(input_width, 3)?, mul(output_width, 2)?)?,
                family.rows,
            )?,
            8,
        )?;
        let mut bytes = add(mul(trace, 5)?, panels)?;
        if self.attention_width > 0 {
            bytes = add(bytes, mul(mul(rows, rows)?, 8 * 12)?)?;
            bytes = add(bytes, mul(mul(rows, self.attention_width)?, 8 * 16)?)?;
        }
        if self.rotary_width > 0 {
            let span = family
                .layout
                .as_ref()
                .and_then(|l| l.position.iter().max())
                .copied()
                .map_or(Ok(0), |p| add(p as usize, 1))?
                .max(self.largest_rotary_span.get());
            bytes = add(
                bytes,
                mul(mul(add(span, rows)?, self.rotary_width)?, 8 * 4)?,
            )?;
        }
        Ok(bytes)
    }

    /// Preserve the complete supplied contexts/layout; no row sampling or token shuffling.
    /// Only final ordered input panels and concatenated target panels cross to the host.
    pub fn capture(
        &self,
        family: &FamilyInputs,
        trace_and_panel_bytes: usize,
    ) -> Result<Panel, String> {
        if family.slots.len() != self.source_slots.len() {
            return Err("local supervision needs one value set per declared slot".into());
        }
        // Check actual upload dimensions before relying on the row-based budget.
        for (slot, (declared, values)) in self.source_slots.iter().zip(&family.slots).enumerate() {
            let valid = match (declared, values) {
                (Slot::Raw { width }, SlotValues::Raw(values)) => values.dim() == (family.rows, *width),
                (Slot::Token { .. }, SlotValues::Tokens(values)) => values.len() == family.rows,
                _ => false,
            };
            if !valid {
                return Err(format!("local supervision slot {slot} values differ from declared type or family shape"));
            }
        }
        // Device attention treats contiguous blocks as independent sequences. A
        // repeated ID in a later block would silently discard earlier context.
        // The structural driver groups whole sequences and scatters rows back;
        // direct callers must provide that same contiguous representation.
        if self.attention_width > 0 {
            if let Some(layout) = &family.layout {
                let mut seen = BTreeSet::new();
                let mut previous = None;
                for &sequence in &layout.sequence {
                    if previous != Some(sequence) && !seen.insert(sequence) {
                        return Err("device local supervision requires each sequence in one contiguous block".into());
                    }
                    previous = Some(sequence);
                }
            }
        }
        let planned = self.planned_trace_and_panel_bytes(family)?;
        if planned > trace_and_panel_bytes {
            return Err(format!(
                "device local supervision trace/panel plan {planned} exceeds {trace_and_panel_bytes}"
            ));
        }
        // Compiled programs retain their latest prepared family and rotary tables. Account
        // for earlier larger families even when a later capture uses fewer rows/positions.
        self.largest_rows
            .set(self.largest_rows.get().max(family.rows));
        if let Some(position) = family.layout.as_ref().and_then(|l| l.position.iter().max()) {
            self.largest_rotary_span.set(
                self.largest_rotary_span
                    .get()
                    .max(add(*position as usize, 1)?),
            );
        }
        let d = self.candidate.device();
        let trace = self.candidate.forward(family)?;
        let all_inputs = trace.value(self.candidate.hidden())?;
        let mut offset = 0;
        let mut resident_inputs = Vec::<Tensor>::new();
        let mut inputs = Vec::new();
        for &width in &self.input_widths {
            let end = add(offset, width)?;
            let value = d
                .columns_of(all_inputs, offset..end)
                .map_err(|e| e.to_string())?;
            let host = d.download(&value).map_err(|e| e.to_string())?;
            if host.dim() != (family.rows, width) || host.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite or incorrectly shaped candidate local inputs".into());
            }
            resident_inputs.push(value);
            inputs.push(host);
            offset = end;
        }
        drop(trace);
        let boundaries = self
            .teacher_boundaries
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        let native_trace = self.teacher.forward_edited(
            family,
            BTreeMap::new(),
            &boundaries,
            |_, _| Ok(()),
            |n, _| {
                self.teacher_boundaries
                    .get(&n)
                    .map(|&i| d.copy(&resident_inputs[i]).map_err(|e| e.to_string()))
                    .transpose()
            },
        )?;
        let targets = d
            .download(native_trace.value(self.teacher.hidden())?)
            .map_err(|e| e.to_string())?;
        if targets.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite same-input native local targets".into());
        }
        Ok(Panel {
            inputs,
            targets,
            report: Report {
                rows: family.rows,
                arguments: self.arguments.clone(),
                native_outputs: self.outputs.clone(),
                output_widths: self.output_widths.clone(),
                native_evaluated_nodes: self.native_evaluated_nodes.clone(),
                native_prefix_nodes_avoided: self.native_prefix_nodes_avoided,
                planned_trace_and_panel_bytes: planned,
                scope: "Prepared F64 device execution; native dependency slice clamped to candidate-reached boundaries. Original boundary producers and unrelated native nodes are removed. Complete supplied sequence layout retained. Downloads only final input/target panels. Operator buffers separately budgeted; trace/panel plan excludes caller-retained panels/inputs, host coefficients, code/index arrays and backend/allocator scratch. No CPU fallback or F32 throughput claim.",
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Declarations, Domain, Law, Scale, SequenceLayout};
    use ndarray::array;

    #[test]
    fn rejects_interleaved_attention_context_and_preserves_contiguous_sequences() {
        let native = OperatorProgram {
            declarations: Declarations {
                domains: vec![], slots: vec![Slot::Raw { width: 1 }; 3], parameters: 0,
            },
            bases: vec![], rules: vec![], operators: vec![],
            nodes: vec![Node::Raw { slot: 0 }, Node::Raw { slot: 1 }, Node::Raw { slot: 2 },
                Node::Attend { query: 0, key: 1, value: 2, scale: Scale::One,
                    rotary: None, causal: true }],
            output: 3,
        };
        let candidate = Artifact::native(&native).unwrap();
        let family = FamilyInputs {
            rows: 4,
            slots: vec![SlotValues::Raw(array![[0.], [0.], [0.], [0.]]),
                SlotValues::Raw(array![[0.], [0.], [0.], [0.]]),
                SlotValues::Raw(array![[10.], [100.], [20.], [200.]])],
            layout: Some(SequenceLayout { sequence: vec![7, 3, 7, 3], position: vec![0, 0, 1, 1] }),
        };
        let prepared = PreparedDeviceCapture::new(&Device::host(), &native, &candidate,
            &[0, 1, 2], &[3], 1 << 20).unwrap();
        let error = prepared.capture(&family, 1 << 20).err().expect("interleaved context must fail");
        assert!(error.contains("one contiguous block"), "{error}");
        let contiguous = family.select(&[0, 2, 1, 3]);
        let actual = prepared.capture(&contiguous, 1 << 20).unwrap();
        let expected = super::super::capture(&native, &candidate, &[0, 1, 2], &[3],
            &contiguous, 1 << 20).unwrap();
        assert_eq!(actual.targets, expected.targets);
        assert_eq!(actual.targets, array![[10.], [15.], [100.], [150.]]);
    }

    #[test]
    fn rejects_malformed_slots_before_budgeting_or_upload() {
        let native = OperatorProgram {
            declarations: Declarations {
                domains: vec![Domain { size: 2 }],
                slots: vec![Slot::Raw { width: 1 }, Slot::Token { domain: 0 }],
                parameters: 0,
            },
            bases: vec![], rules: vec![], operators: vec![],
            nodes: vec![Node::Raw { slot: 0 }, Node::Pointwise { input: 0, laws: vec![Law::Relu] }],
            output: 1,
        };
        let candidate = Artifact::native(&native).unwrap();
        let prepared = PreparedDeviceCapture::new(&Device::host(), &native, &candidate,
            &[0], &[1], 1 << 20).unwrap();
        let valid = FamilyInputs { rows: 1, slots: vec![SlotValues::Raw(array![[2.]]),
            SlotValues::Tokens(vec![0])], layout: None };
        for (slot, malformed) in [
            (0, SlotValues::Raw(array![[2.], [3.]])),
            (0, SlotValues::Raw(array![[2., 3.]])),
            (0, SlotValues::Tokens(vec![0])),
            (1, SlotValues::Tokens(vec![0, 1])),
            (1, SlotValues::Raw(array![[2.]])),
        ] {
            let mut family = valid.clone();
            family.slots[slot] = malformed;
            let error = prepared.capture(&family, 0).err().expect("invalid family must fail");
            assert!(error.contains(&format!("slot {slot} values differ")), "{error}");
        }
        assert_eq!(prepared.capture(&valid, 1 << 20).unwrap().targets, array![[2.]]);
    }
}
