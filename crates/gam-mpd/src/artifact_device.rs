//! Device execution of decoded artifacts, preserving root-node identity and exception order.
use crate::operator_program::{Node, Operator, OperatorProgram, remap_node};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_program::DeviceProgram;
    use gam_gpu::tensor::Device;
    use ndarray::Array2;
    use std::collections::BTreeMap;

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
