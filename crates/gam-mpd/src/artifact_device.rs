//! Device execution of decoded artifacts, preserving root-node identity and exception order.
use crate::artifact::{Artifact, contexts};
use crate::device_program::{DeviceProgram, DeviceTrace};
use crate::operator_program::{FamilyInputs, Node, Operator, OperatorProgram, remap_node};
use gam_gpu::tensor::{Device, Tensor};
use ndarray::Array2;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Each old root node has a distinct materialized value, including identity-call outputs.
pub fn mapped_inlined(program: &OperatorProgram) -> Result<(OperatorProgram, Vec<usize>), String> {
    let (flat, roots, _) = mapped_inlined_observed(program, &[])?;
    Ok((flat, roots))
}

/// Observe actual invocation values without adding arithmetic or invented native
/// identities. A path starts with a root-node index; every further index selects
/// a node in the preceding Call's rule body. A terminal Call names its existing
/// materialized output barrier; a Param names its bound actual input. Results
/// retain requested order, including duplicate paths. Invalid paths are refused.
pub fn mapped_inlined_observed(
    program: &OperatorProgram,
    paths: &[Vec<usize>],
) -> Result<(OperatorProgram, Vec<usize>, Vec<usize>), String> {
    program.interfaces().map_err(|e| e.to_string())?;
    if paths.iter().any(Vec::is_empty) {
        return Err("observation paths require a root node".into());
    }
    fn expand(
        original: &OperatorProgram,
        body: &[Node],
        args: &[usize],
        flat: &mut OperatorProgram,
        requests: &[(usize, &[usize])],
        observed: &mut [usize],
    ) -> Result<Vec<usize>, String> {
        if requests
            .iter()
            .any(|(_, path)| path.first().is_none_or(|node| *node >= body.len()))
        {
            return Err("observation node outside root or invoked rule body".into());
        }
        let mut map = Vec::with_capacity(body.len());
        for (index, node) in body.iter().enumerate() {
            let nested = requests
                .iter()
                .filter(|(_, path)| path[0] == index && path.len() > 1)
                .map(|(id, path)| (*id, &path[1..]))
                .collect::<Vec<_>>();
            if !nested.is_empty() && !matches!(node, Node::Call { .. }) {
                return Err("observation path descends through a non-Call node".into());
            }
            let value = match node {
                Node::Param { index } => *args.get(*index).ok_or("unbound rule parameter")?,
                Node::Call { rule, arguments } => {
                    let rule = original.rules.get(*rule).ok_or("missing rule")?;
                    let args: Vec<_> = arguments.iter().map(|&i| map[i]).collect();
                    let inner = expand(original, &rule.nodes, &args, flat, &nested, observed)?;
                    let source = inner[rule.output];
                    // Same barrier as the unobserved expansion, preserving alias
                    // separation for exceptions/interventions on this invocation.
                    flat.output = source;
                    let interface = flat.node_interface(source).map_err(|e| e.to_string())?;
                    let operator = flat.operators.len();
                    flat.operators.push(Arc::new(Operator::identity(
                        "call output barrier",
                        interface,
                    )));
                    flat.nodes.push(Node::Affine {
                        terms: vec![(source, operator)],
                        bias: None,
                    });
                    flat.nodes.len() - 1
                }
                other => {
                    let mut copy = other.clone();
                    let ops: Vec<_> = (0..flat.operators.len()).collect();
                    let bases: Vec<_> = (0..flat.bases.len()).collect();
                    let rules: Vec<_> = (0..original.rules.len()).collect();
                    remap_node(&mut copy, &map, &ops, &bases, &rules);
                    flat.nodes.push(copy);
                    flat.nodes.len() - 1
                }
            };
            for (id, path) in requests {
                if path[0] == index && path.len() == 1 {
                    observed[*id] = value;
                }
            }
            map.push(value);
        }
        Ok(map)
    }
    let mut flat = program.clone();
    flat.nodes.clear();
    let requests = paths
        .iter()
        .enumerate()
        .map(|(id, path)| (id, path.as_slice()))
        .collect::<Vec<_>>();
    let mut observed = vec![usize::MAX; paths.len()];
    let map = expand(
        program,
        &program.nodes,
        &[],
        &mut flat,
        &requests,
        &mut observed,
    )?;
    if observed.contains(&usize::MAX) {
        return Err("observation path was not materialized".into());
    }
    flat.output = map[program.output];
    flat.rules.clear();
    flat.interfaces().map_err(|e| e.to_string())?;
    Ok((flat, map, observed))
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
                (Some(base), Some(limit)) => DeviceProgram::compile_values_sharing_bounded(&base.program, &flat, limit)?,
                (Some(base), None) => DeviceProgram::compile_values_sharing(&base.program, &flat)?,
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
    /// Incomplete resident-value memory estimate. Excludes attention workspaces, weights,
    /// exception tensors and allocation overhead. The caller chooses and limits batch rows.
    pub fn estimated_resident_bytes(&self, rows: usize) -> Result<usize, String> {
        rows.checked_mul(self.program.edited_bytes_per_row()).ok_or("edited batch size overflow".into())
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
        let excepted: std::collections::BTreeSet<usize> = tensors.keys().copied().collect();
        if materialize_head {
            self.program.forward_edited(family, BTreeMap::new(), &excepted, before, after)
        } else {
            self.program.forward_edited_intermediates(family, &excepted, before, after)
        }
    }

    /// Borrow a materialized output for resident reductions without copying its
    /// device buffer. The trace owns the buffer for the duration of the borrow.
    pub fn output_ref<'a>(&self, trace: &'a DeviceTrace) -> Result<&'a Tensor, String> {
        trace.value(self.output)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    
    
    #[test]
    fn four_layer_terminal_readout_materializes_after_logit_exception_before_readout_edit() {
        let dir = crate::test_support::tiny_export("artifact_device_readout", 4);
        let imported = crate::import::import_language_model(&dir, 2, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let program = imported.program;
        let family = imported.family;
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
                &[logits].into(),
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
        let dir = crate::test_support::tiny_export("device_intermediate_head", 2);
        let imported = crate::import::import_language_model(&dir, 1, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let p = imported.program;
        let family = imported.family;
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
                &Default::default(),
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

}
