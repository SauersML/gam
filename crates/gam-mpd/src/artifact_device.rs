//! Device execution of decoded artifacts, preserving root-node identity and exception order.
use crate::artifact::{Artifact, EncodedArtifact, contexts};
use crate::device_program::{DeviceProgram, DeviceTrace};
use crate::operator_program::{FamilyInputs, Node, Operator, OperatorProgram, remap_node};
use crate::precision::DecodableArtifact;
use gam_gpu::tensor::{Device, Tensor};
use ndarray::Array2;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Each old root node has a distinct materialized value, including identity-call outputs.
pub fn mapped_inlined(program: &OperatorProgram) -> Result<(OperatorProgram, Vec<usize>), String> {
    program.interfaces().map_err(|e| e.to_string())?;
    fn expand(original: &OperatorProgram, body: &[Node], args: &[usize], flat: &mut OperatorProgram) -> Result<Vec<usize>, String> {
        let mut map = Vec::with_capacity(body.len());
        for node in body {
            let value = match node {
                Node::Param { index } => *args.get(*index).ok_or("unbound rule parameter")?,
                Node::Call { rule, arguments } => {
                    let rule = original.rules.get(*rule).ok_or("missing rule")?;
                    let args: Vec<usize> = arguments.iter().map(|&i| map[i]).collect();
                    let inner = expand(original, &rule.nodes, &args, flat)?;
                    let source = inner[rule.output];
                    // Always copy at the call boundary: an aliasing Param cannot expose its
                    // parent's allocation to exceptions/interventions on this call's result.
                    flat.output = source;
                    let interface = flat.node_interface(source).map_err(|e| e.to_string())?;
                    let operator = flat.operators.len();
                    flat.operators.push(Arc::new(Operator::identity("call output barrier", interface)));
                    flat.nodes.push(Node::Affine { terms: vec![(source, operator)], bias: None });
                    flat.nodes.len() - 1
                }
                other => {
                    let mut copy = other.clone();
                    let ops: Vec<usize> = (0..flat.operators.len()).collect();
                    let bases: Vec<usize> = (0..flat.bases.len()).collect();
                    let rules: Vec<usize> = (0..original.rules.len()).collect();
                    remap_node(&mut copy, &map, &ops, &bases, &rules);
                    flat.nodes.push(copy);
                    flat.nodes.len() - 1
                }
            };
            map.push(value);
        }
        Ok(map)
    }
    let mut flat = program.clone();
    flat.nodes.clear();
    let map = expand(program, &program.nodes, &[], &mut flat)?;
    flat.output = map[program.output];
    flat.rules.clear();
    flat.interfaces().map_err(|e| e.to_string())?;
    Ok((flat, map))
}

/// Expand a decoded artifact's program and every executable/binding root-node reference.
/// Original operator indices and decoded derived values are preserved; barriers append operators.
pub fn expanded_artifact(artifact: &Artifact) -> Result<(Artifact, Vec<usize>), String> {
    let (program, map) = mapped_inlined(&artifact.program)?;
    let mut expanded = artifact.clone();
    expanded.program = program;
    for (_, node) in &mut expanded.places {
        *node = map[*node];
    }
    for block in &mut expanded.blocks {
        block.write = map[block.write];
        for node in &mut block.reads {
            *node = map[*node];
        }
    }
    for exception in &mut expanded.exceptions {
        exception.node = map[exception.node];
    }
    Ok((expanded, map))
}

pub struct Resident {
    artifact: Artifact,
    /// Root old -> flat. Internal rule nodes have no root edit identity.
    map: Vec<usize>,
    roots: BTreeMap<usize, usize>,
    program: DeviceProgram,
    output: usize,
}
impl Resident {
    /// Refuse unsupported primitives and a host device explicitly. No execution falls back.
    /// Encode/decode first: derived operators are reconstructed by the artifact decoder.
    pub fn new(device: &Device, candidate: &Artifact) -> Result<Self, String> {
        if device.is_host() {
            return Err("artifact device needs an accelerator".into());
        }
        let artifact = EncodedArtifact::of(&candidate.f32_literals()?)?.decode()?;
        Self::from_decoded(device, &artifact)
    }
    /// Acceptance already supplies decoded P. Preserve its executed numerical values;
    /// do not repeat literal rounding, serialization or derived-operator computation.
    pub fn from_decoded(device: &Device, candidate: &Artifact) -> Result<Self, String> {
        if device.is_host() || !device.float64() {
            return Err("artifact device needs a float64 accelerator".into());
        }
        Self::compile_decoded(device, candidate, None)
    }
    /// Execute an already decoded artifact's actual value output, including a
    /// multi-term affine or Concat. No synthetic head or additional arithmetic.
    /// Refuses unsupported device nodes and host fallback explicitly.
    pub fn from_decoded_values(device: &Device, candidate: &Artifact) -> Result<Self, String> {
        if device.is_host() || !device.float64() {
            return Err("artifact device needs a float64 accelerator".into());
        }
        Self::compile_decoded_mode(device, candidate, None, true)
    }
    /// Value execution sharing only exact operator Arcs in identical roles.
    pub fn from_decoded_values_sharing(from: &Self, candidate: &Artifact) -> Result<Self, String> {
        if from.program.device().is_host() || !from.program.device().float64() {
            return Err("artifact device needs a float64 accelerator".into());
        }
        Self::compile_decoded_mode(from.program.device(), candidate, Some(from), true)
    }
    /// Preserve supplied values, bounding retained numeric source operators before upload.
    pub fn from_decoded_values_bounded(device: &Device, candidate: &Artifact, numeric_bytes_limit: usize) -> Result<Self, String> {
        if device.is_host() || !device.float64() { return Err("artifact device needs a float64 accelerator".into()); }
        Self::compile_decoded_mode_bounded(device, candidate, None, true, Some(numeric_bytes_limit))
    }
    pub fn operator_numeric_bytes(&self) -> Result<usize, String> { self.program.operator_numeric_bytes() }
    /// Share only identical operator Arcs in identical executable roles with this
    /// candidate's base resident; graph edits and root mappings remain independent.
    pub fn from_decoded_sharing(from: &Self, candidate: &Artifact) -> Result<Self, String> {
        if from.program.device().is_host() || !from.program.device().float64() {
            return Err("artifact device needs a float64 accelerator".into());
        }
        Self::compile_decoded(from.program.device(), candidate, Some(from))
    }
    fn compile_decoded(device: &Device, candidate: &Artifact, from: Option<&Self>) -> Result<Self, String> {
        Self::compile_decoded_mode(device, candidate, from, false)
    }
    fn compile_decoded_mode(device: &Device, candidate: &Artifact, from: Option<&Self>, values: bool) -> Result<Self, String> {
        Self::compile_decoded_mode_bounded(device, candidate, from, values, None)
    }
    fn compile_decoded_mode_bounded(device: &Device, candidate: &Artifact, from: Option<&Self>, values: bool, numeric_bytes_limit: Option<usize>) -> Result<Self, String> {
        candidate.program.interfaces().map_err(|e| e.to_string())?;
        let artifact = candidate.clone();
        let (flat, map) = mapped_inlined(&artifact.program)?;
        let output = flat.output;
        for exception in &artifact.exceptions {
            let node = artifact.program.nodes.get(exception.node).ok_or("exception node outside artifact")?;
            let width = artifact.program.node_interface(exception.node).map_err(|e| e.to_string())?.width();
            if exception.column >= width || !exception.value.is_finite() {
                return Err("invalid exception column or nonfinite value".into());
            }
            if matches!(node, Node::Feature { .. }) {
                return Err("artifact feature-node exceptions need unsupported token-basis materialization".into());
            }
        }
        let program = if values {
            match (from, numeric_bytes_limit) {
                (Some(base), _) => DeviceProgram::compile_values_sharing(&base.program, &flat)?,
                (None, Some(limit)) => DeviceProgram::compile_values_bounded(device, &flat, limit)?,
                (None, None) => DeviceProgram::compile_values(device, &flat)?,
            }
        } else {
            match from {
                Some(base) => DeviceProgram::compile_sharing(&base.program, &flat)?,
                None => DeviceProgram::compile(device, &flat)?,
            }
        };
        let roots = map.iter().enumerate().map(|(old, &new)| (new, old)).collect();
        Ok(Self { artifact, map, roots, program, output })
    }
    pub fn place(&self, native: usize) -> Option<usize> {
        self.artifact.place(native).map(|old| self.map[old])
    }
    pub fn root_value<'a>(&self, trace: &'a DeviceTrace, root: usize) -> Result<&'a Tensor, String> {
        let node = *self.map.get(root).ok_or("root node outside artifact")?;
        trace.value(node)
    }
    /// Incomplete resident-value memory estimate. Excludes attention workspaces, weights,
    /// exception tensors and allocation overhead. The caller chooses and limits batch rows.
    pub fn estimated_resident_bytes(&self, rows: usize) -> Result<usize, String> {
        rows.checked_mul(self.program.edited_bytes_per_row()).ok_or("edited batch size overflow".into())
    }
    /// Values retained by intermediate execution; excludes weights, attention workspaces,
    /// edit masks, exceptions and allocation overhead.
    pub fn estimated_intermediate_bytes(&self, rows: usize) -> Result<usize, String> {
        rows.checked_mul(self.program.bytes_per_row()).ok_or("intermediate batch size overflow".into())
    }
    /// Add exceptions, then invoke root-node edits. Edits return a fresh replacement tensor;
    /// they cannot mutate an aliased upstream value. DeviceTrace indices here are expanded.
    pub fn forward_edited(
        &self,
        family: &FamilyInputs,
        edit: impl FnMut(usize, &DeviceTrace) -> Result<Option<Tensor>, String>,
    ) -> Result<DeviceTrace, String> {
        self.forward_hooks(family, true, edit)
    }
    /// Preserve root edit identity while leaving the checked native head streamed.
    pub fn forward_edited_intermediates(
        &self,
        family: &FamilyInputs,
        edit: impl FnMut(usize, &DeviceTrace) -> Result<Option<Tensor>, String>,
    ) -> Result<DeviceTrace, String> {
        if self.artifact.exceptions.iter().any(|e| self.program.is_streamed_head(self.map[e.node])) {
            return Err("intermediate execution cannot skip an executable head exception".into());
        }
        self.forward_hooks(family, false, edit)
    }
    fn forward_hooks(
        &self,
        family: &FamilyInputs,
        materialize_head: bool,
        mut edit: impl FnMut(usize, &DeviceTrace) -> Result<Option<Tensor>, String>,
    ) -> Result<DeviceTrace, String> {
        let d = self.program.device();
        // Each addition must round against the live value in serialized exception
        // order. Summing overlapping additions first changes cancellation semantics.
        let mut tensors: BTreeMap<usize, Vec<Tensor>> = BTreeMap::new();
        let row_contexts = contexts(family);
        for exception in &self.artifact.exceptions {
            let node = self.map[exception.node];
            let width = self.artifact.program.node_interface(exception.node).map_err(|e| e.to_string())?.width();
            let mut values = Array2::zeros((family.rows, width));
            let mut matched = false;
            for (row, context) in row_contexts.iter().enumerate() {
                if *context == exception.context {
                    values[[row, exception.column]] = f64::from(exception.value);
                    matched = true;
                }
            }
            if matched {
                tensors.entry(node).or_default().push(d.upload(values.view()).map_err(|e| e.to_string())?);
            }
        }
        // Two hooks are required: exception addition is visible in the trace before edit runs.
        let before = |node, value: &mut Tensor| {
            for add in tensors.get(&node).map_or(&[][..], Vec::as_slice) {
                d.axpy(value, 1.0, add).map_err(|e| e.to_string())?;
            }
            Ok(())
        };
        let after = |node, trace: &DeviceTrace| match self.roots.get(&node) {
            Some(&root) => edit(root, trace),
            None => Ok(None),
        };
        if materialize_head {
            self.program.forward_edited(family, BTreeMap::new(), before, after)
        } else {
            self.program.forward_edited_intermediates(family, before, after)
        }
    }

    pub fn output(&self, trace: &DeviceTrace) -> Result<Tensor, String> {
        self.program.device().copy(trace.value(self.output)?).map_err(|e| e.to_string())
    }

    /// Borrow a materialized output for resident reductions without copying its
    /// device buffer. The trace owns the buffer for the duration of the borrow.
    pub fn output_ref<'a>(&self, trace: &'a DeviceTrace) -> Result<&'a Tensor, String> {
        trace.value(self.output)
    }

    /// Whether a root intervention would hit an output left unmaterialized by
    /// intermediate execution. Such an intervention must not be silently skipped.
    pub fn is_streamed_head_root(&self, root: usize) -> Result<bool, String> {
        let node = *self.map.get(root).ok_or("root node outside artifact")?;
        Ok(self.program.is_streamed_head(node))
    }

    /// Bounded caller-selected tile of this artifact's own compiled output head.
    pub fn logits_rows(&self, trace: &DeviceTrace, start: usize, rows: usize) -> Result<Tensor, String> {
        self.program.logits_rows(trace, start, rows)
    }

    /// Hidden width and vocabulary width for sizing a streamed output tile.
    pub fn streamed_head_dimensions(&self) -> (usize, usize) {
        (self.program.widths()[self.program.hidden()], self.program.classes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{Exception, OperatorLaw};
    use crate::operator_program::{Declarations, Interface, Rule, Slot, SlotValues};
    fn fixture(nested: bool) -> (Artifact, FamilyInputs) {
        let interface = Interface::native(2).unwrap();
        let mut rules = vec![Rule { name: "identity".into(), inputs: vec![interface.clone()], nodes: vec![Node::Param { index: 0 }], output: 0 }];
        if nested {
            rules.push(Rule {
                name: "nested identity".into(),
                inputs: vec![interface.clone()],
                nodes: vec![Node::Param { index: 0 }, Node::Call { rule: 0, arguments: vec![0] }],
                output: 1,
            });
        }
        let program = OperatorProgram {
            declarations: Declarations { domains: Vec::new(), slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
            bases: Vec::new(),
            operators: vec![Arc::new(Operator::identity("I", interface))],
            rules,
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Call { rule: usize::from(nested), arguments: vec![0] },
                Node::Affine { terms: vec![(0, 0)], bias: None },
                Node::Affine { terms: vec![(1, 0), (2, 0)], bias: None },
            ],
            output: 3,
        };
        (Artifact::native(&program).unwrap(), FamilyInputs { rows: 1, slots: vec![SlotValues::Raw(ndarray::array![[1.0, 2.0]])], layout: None })
    }
    fn alias_test(nested: bool) {
        let (mut artifact, family) = fixture(nested);
        artifact.exceptions.push(Exception { context: vec![], node: 1, column: 0, value: 3.0 });
        let original = artifact
            .execute_edited(&family, |node, values, _| {
                if node == 1 {
                    values.mapv_inplace(|v| 2.0 * v);
                }
                Ok(())
            })
            .unwrap();
        let (expanded, map) = expanded_artifact(&artifact).unwrap();
        assert_ne!(map[0], map[1]);
        let mut callbacks = Vec::new();
        let trace = expanded
            .execute_edited(&family, |node, values, _| {
                if let Some(old) = map.iter().position(|&new| new == node) {
                    callbacks.push(old);
                    if old == 1 {
                        values.mapv_inplace(|v| 2.0 * v);
                    }
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(callbacks, vec![0, 1, 2, 3]);
        assert_eq!(trace.values[map[0]], ndarray::array![[1.0, 2.0]]);
        assert_eq!(trace.values[map[2]], ndarray::array![[1.0, 2.0]]);
        assert_eq!(trace.values[expanded.program.output], ndarray::array![[9.0, 6.0]]);
        assert_eq!(trace.values[expanded.program.output], original.values[artifact.program.output]);
    }
    #[test]
    fn identity_call_exception_and_intervention_do_not_change_input_or_sibling() {
        alias_test(false);
    }
    #[test]
    fn nested_identity_calls_have_isolated_output_barriers_and_root_order() {
        alias_test(true);
    }
    #[test]
    fn already_decoded_derived_operator_values_survive_expansion() {
        let (mut artifact, family) = fixture(true);
        let interface = Interface::native(2).unwrap();
        for name in ["gain", "final gain", "derived"] {
            artifact.program.operators.push(Arc::new(Operator::identity(name, interface.clone())));
        }
        artifact = artifact.derive(3, OperatorLaw::Copy { value: 0, gain: 1, final_gain: 2 }, 0.5, vec![]).unwrap();
        artifact.program.nodes[2] = Node::Affine { terms: vec![(0, 3)], bias: None };
        let decoded = EncodedArtifact::of(&artifact).unwrap().decode().unwrap();
        let (expanded, _) = expanded_artifact(&decoded).unwrap();
        assert!(Arc::ptr_eq(&decoded.program.operators[3], &expanded.program.operators[3]));
        assert_eq!(expanded.derived, decoded.derived);
        let a = decoded.execute(&family).unwrap();
        let b = expanded.execute(&family).unwrap();
        assert_eq!(a.values[decoded.program.output], b.values[expanded.program.output]);
    }
    #[test]
    fn four_layer_terminal_readout_materializes_after_logit_exception_before_readout_edit() {
        let dir = crate::explanation_tests::tiny_export("artifact_device_readout", 4);
        let imported = crate::import::import_language_model(&dir, 2, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let program = imported.program;
        let family = imported.contract.family;
        let Node::Readout { input: logits, .. } = program.nodes[program.output] else {
            panic!("fixture must exercise the actual terminal Readout architecture");
        };
        let device = Device::host();
        let resident = DeviceProgram::compile(&device, &program).unwrap();
        let expected = program
            .execute_edited(&family, |node, values, _| {
                if node == logits {
                    values[[0, 0]] += 3.0;
                }
                if node == program.output {
                    values.mapv_inplace(|v| 2.0 * v);
                }
                Ok(())
            })
            .unwrap();
        let width = program.node_interface(logits).unwrap().width();
        let mut add = Array2::zeros((family.rows, width));
        add[[0, 0]] = 3.0;
        let add = device.upload(add.view()).unwrap();
        let actual = resident
            .forward_edited(
                &family,
                BTreeMap::new(),
                |node, value| {
                    if node == logits {
                        device.axpy(value, 1.0, &add).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                },
                |node, trace| {
                    if node == program.output {
                        let mut doubled = device.copy(trace.value(node)?).map_err(|e| e.to_string())?;
                        device.axpy(&mut doubled, 1.0, trace.value(node)?).map_err(|e| e.to_string())?;
                        Ok(Some(doubled))
                    } else {
                        Ok(None)
                    }
                },
            )
            .unwrap();
        let output = device.download(actual.value(program.output).unwrap()).unwrap();
        let gap = (&output - &expected.values[program.output]).iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(gap < 1e-10, "terminal readout edit gap {gap}");
        assert!(actual.value(logits).is_ok());
        assert!(actual.value(program.output).is_ok());
    }
    #[test]
    fn edited_intermediates_skip_real_native_head_but_preserve_residual_edits() {
        let dir = crate::explanation_tests::tiny_export("device_intermediate_head", 2);
        let imported = crate::import::import_language_model(&dir, 1, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let p = imported.program;
        let family = imported.contract.family;
        let last = p
            .nodes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, n)| matches!(n, Node::Affine { terms, .. } if terms.len() == 2).then_some(i))
            .expect("final residual Add");
        let device = Device::host();
        let resident = DeviceProgram::compile(&device, &p).unwrap();
        let expected = p
            .execute_edited(&family, |node, value, _| {
                if node == last {
                    value.mapv_inplace(|v| v * 0.7);
                }
                Ok(())
            })
            .unwrap();
        let actual = resident
            .forward_edited_intermediates(
                &family,
                |_, _| Ok(()),
                |node, trace| {
                    if node != last {
                        return Ok(None);
                    }
                    let source = trace.value(node)?;
                    let mut value = device.copy(source).map_err(|e| e.to_string())?;
                    let scale = device.upload(Array2::from_elem((source.rows(), source.cols()), 0.7).view()).map_err(|e| e.to_string())?;
                    device.hadamard(&mut value, source, &scale, false).map_err(|e| e.to_string())?;
                    Ok(Some(value))
                },
            )
            .unwrap();
        let actual_residual = device.download(actual.value(last).unwrap()).unwrap();
        for (a, b) in actual_residual.iter().zip(expected.values[last].iter()) {
            assert!((a - b).abs() < 1e-10, "residual differs: {a} versus {b}");
        }
        assert!(actual.value(p.output).is_err(), "native head/readout must stay unmaterialized");
    }
    #[test]
    fn overlapping_exceptions_round_in_order_before_intermediate_edit() {
        let dir = crate::explanation_tests::tiny_export("ordered_device_exceptions", 2);
        let imported = crate::import::import_language_model(&dir, 1, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let mut artifact = Artifact::native(&imported.program).unwrap();
        let family = imported.contract.family;
        let root = artifact
            .program
            .nodes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, n)| matches!(n, Node::Affine { terms, .. } if terms.len() == 2).then_some(i))
            .unwrap();
        let context = contexts(&family)[0].clone();
        for value in [1e20f32, -1e20f32, 0.25f32] {
            artifact.exceptions.push(Exception { context: context.clone(), node: root, column: 0, value });
        }
        let expected = artifact
            .execute_edited(&family, |node, values, _| {
                if node == root {
                    values.mapv_inplace(|v| v * 2.0);
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(expected.values[root][[0, 0]], 0.5);
        let device = Device::host();
        // Reference tensor test bypasses only the production accelerator requirement.
        let (flat, map) = mapped_inlined(&artifact.program).unwrap();
        let output = flat.output;
        let roots = map.iter().enumerate().map(|(old, &new)| (new, old)).collect();
        let program = DeviceProgram::compile(&device, &flat).unwrap();
        let resident = Resident { artifact, map, roots, program, output };
        for materialize_head in [false, true] {
            let actual = resident
                .forward_hooks(&family, materialize_head, |node, trace| {
                    if node != root {
                        return Ok(None);
                    }
                    let source = resident.root_value(trace, root)?;
                    let mut doubled = device.copy(source).map_err(|e| e.to_string())?;
                    device.axpy(&mut doubled, 1.0, source).map_err(|e| e.to_string())?;
                    Ok(Some(doubled))
                })
                .unwrap();
            let values = device.download(resident.root_value(&actual, root).unwrap()).unwrap();
            assert_eq!(values[[0, 0]], 0.5);
            for (a, b) in values.iter().zip(expected.values[root].iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn value_mode_materializes_genuine_language_readout_without_synthetic_head() {
        let dir = crate::explanation_tests::tiny_export("resident_value_readout", 2);
        let imported = crate::import::import_language_model(&dir, 1, 6).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let device = Device::host();
        let artifact = Artifact::native(&imported.program).unwrap();
        let streamed = Resident::compile_decoded(&device, &artifact, None).unwrap();
        let values = Resident::compile_decoded_mode(&device, &artifact, None, true).unwrap();
        assert_eq!(values.artifact.program.operators.len(), artifact.program.operators.len());
        let a = streamed.program.forward(&imported.contract.family).unwrap();
        let b = values.forward_edited(&imported.contract.family, |_, _| Ok(None)).unwrap();
        let expected = device.download(&streamed.program.logits_on_device(&a).unwrap()).unwrap();
        let actual = device.download(&values.output(&b).unwrap()).unwrap();
        assert_eq!(actual, expected);
        assert!(!values.program.is_streamed_head(values.output));
    }

    #[test]
    fn value_mode_executes_local_multiblock_concat_and_ordered_exceptions() {
        let (mut artifact, family) = fixture(true);
        let native = artifact.program.clone();
        artifact.blocks = vec![
            crate::artifact::Binding { name: "call".into(), native_reads: vec![0], native_write: 1, reads: vec![0], write: 1 },
            crate::artifact::Binding { name: "sum".into(), native_reads: vec![0], native_write: 3, reads: vec![0], write: 3 },
        ];
        for value in [1e20f32, -1e20f32, 0.375f32] {
            artifact.exceptions.push(Exception { context: vec![], node: 1, column: 0, value });
        }
        let (local, columns) = artifact.local_artifact(&native).unwrap();
        assert_eq!(columns, vec![0..2, 2..4]);
        assert!(matches!(local.program.nodes[local.program.output], Node::Concat { .. }));
        let expected = local.execute(&family).unwrap();
        let device = Device::host();
        assert!(Resident::from_decoded_values(&device, &local).is_err(), "public value execution must refuse host fallback");
        assert!(Resident::compile_decoded(&device, &local, None).is_err(), "LM mode retains the genuine dense-head contract");
        let resident = Resident::compile_decoded_mode(&device, &local, None, true).unwrap();
        let trace = resident.forward_edited(&family, |_, _| Ok(None)).unwrap();
        let actual = device.download(&resident.output(&trace).unwrap()).unwrap();
        assert_eq!(actual, expected.values[local.program.output]);
        let native_artifact = Artifact::native(&native).unwrap();
        let source = Resident::compile_decoded_mode(&device, &native_artifact, None, true).unwrap();
        assert!(Resident::from_decoded_values_sharing(&source, &local).is_err(), "public sharing refuses host fallback");
        let shared = Resident::compile_decoded_mode(&device, &local, Some(&source), true).unwrap();
        let shared_trace = shared.forward_edited(&family, |_, _| Ok(None)).unwrap();
        assert_eq!(device.download(&shared.output(&shared_trace).unwrap()).unwrap(), actual);
        let untouched = source.forward_edited(&family, |_, _| Ok(None)).unwrap();
        assert_eq!(device.download(&source.output(&untouched).unwrap()).unwrap(), native_artifact.execute(&family).unwrap().values[native.output]);
        assert_eq!(actual, ndarray::array![[-0.625, 0.0, -0.625, 0.0]]);
        assert!(!resident.program.is_streamed_head(resident.output));
        assert!(matches!(resident.program.logits_on_device(&trace), Err(error) if error.contains("no linear head")));
        let zeros = device.zeros(family.rows, actual.ncols()).unwrap();
        assert!(resident.program.score_only(&trace, &zeros, None).is_err());
        assert!(resident.program.sampled_many(&trace, &[], None).is_err());
        assert!(resident.program.jvp(&trace, &BTreeMap::new(), gam_gpu::tensor::Arithmetic::F64).is_err());
        assert!(resident.program.vjp(&trace, zeros, &[], gam_gpu::tensor::Arithmetic::F64).is_err());
    }

    #[test]
    fn shared_resident_keeps_independent_root_edits_and_decoded_values() {
        let dir = crate::explanation_tests::tiny_export("shared_device_resident", 2);
        let imported = crate::import::import_language_model(&dir, 1, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let mut artifact = Artifact::native(&imported.program).unwrap();
        for op in &mut artifact.program.operators {
            match &mut Arc::make_mut(op).body {
                crate::operator_program::OperatorBody::Dense { values, present, .. } => {
                    *values = values.as_standard_layout().to_owned();
                    *present = present.as_standard_layout().to_owned();
                }
                crate::operator_program::OperatorBody::LowRank { left, right, .. } => {
                    *left = left.as_standard_layout().to_owned();
                    *right = right.as_standard_layout().to_owned();
                }
                _ => continue
            }
        }
        let artifact = artifact.f32_literals().unwrap();
        let artifact = Artifact::from_bytes(&artifact.to_bytes().unwrap(), &artifact.program.declarations).unwrap();
        let family = imported.contract.family;
        let root = artifact
            .program
            .nodes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, n)| matches!(n, Node::Affine { terms, .. } if terms.len() == 2).then_some(i))
            .unwrap();
        let device = Device::host();
        let base = Resident::compile_decoded(&device, &artifact, None).unwrap();
        assert!(Resident::from_decoded_sharing(&base, &artifact).is_err(), "public sharing must refuse host fallback");
        // A fresh acceptance decode has different Arcs. Reverse operator indices
        // as a compacted candidate can, then recover sharing by exact body identity.
        let mut changed = Artifact::from_bytes(&artifact.to_bytes().unwrap(), &artifact.program.declarations).unwrap();
        let operators: Vec<_> = (0..changed.program.operators.len()).rev().collect();
        changed.program.operators.reverse();
        let nodes: Vec<_> = (0..changed.program.nodes.len()).collect();
        let bases: Vec<_> = (0..changed.program.bases.len()).collect();
        let rules: Vec<_> = (0..changed.program.rules.len()).collect();
        for node in &mut changed.program.nodes {
            crate::operator_program::remap_node(node, &nodes, &operators, &bases, &rules);
        }
        // Decode the permuted wire representation to restore canonical metadata.
        changed = Artifact::from_bytes(&changed.to_bytes().unwrap(), &changed.program.declarations).unwrap();
        let bytes = changed.to_bytes().unwrap();
        assert_eq!(crate::decoded_intern::DecodedOperatorInterner::new(&artifact).unwrap().intern(&mut changed), operators.len());
        assert_eq!(changed.to_bytes().unwrap(), bytes);
        changed.exceptions.push(Exception { context: contexts(&family)[0].clone(), node: root, column: 0, value: 0.375 });
        let expected = changed.execute(&family).unwrap();
        // Test-only tensor reference exercises the same shared compilation path.
        let shared = Resident::compile_decoded(&device, &changed, Some(&base)).unwrap();
        let fresh = Resident::compile_decoded(&device, &changed, None).unwrap();
        assert!(Arc::ptr_eq(&base.artifact.program.operators[0], shared.artifact.program.operators.last().unwrap()));
        let a = shared.forward_edited_intermediates(&family, |_, _| Ok(None)).unwrap();
        let b = fresh.forward_edited_intermediates(&family, |_, _| Ok(None)).unwrap();
        let a = device.download(shared.root_value(&a, root).unwrap()).unwrap();
        let b = device.download(fresh.root_value(&b, root).unwrap()).unwrap();
        assert_eq!(a, b);
        // Candidate exception forwards cannot alter the immutable source tensors.
        let untouched = base.forward_edited_intermediates(&family, |_, _| Ok(None)).unwrap();
        let untouched = device.download(base.root_value(&untouched, root).unwrap()).unwrap();
        let native = artifact.execute(&family).unwrap();
        for (a, b) in untouched.iter().zip(native.values[root].iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        for (a, b) in a.iter().zip(expected.values[root].iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
