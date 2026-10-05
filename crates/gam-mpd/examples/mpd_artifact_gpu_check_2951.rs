//! Explicit CUDA artifact parity check; CUDA absence fails instead of using the host.
//! mpd_artifact_gpu_check_2951 EXPORT [SEQUENCES=1] [CONTEXT=12] [ABS_TOLERANCE=1e-8]
//! Compares a decoded Rule/Call artifact with a context exception and node interventions against
//! its CPU execution. The original language-model logits and terminal Readout are materialized.
use gam_gpu::tensor::Device;
use gam_mpd::artifact::{Artifact, EncodedArtifact, Exception, contexts};
use gam_mpd::artifact_device::Resident;
use gam_mpd::import::import_language_model;
use gam_mpd::operator_program::{Node, Rule};
use gam_mpd::precision::DecodableArtifact;
use serde_json::json;
use std::path::Path;

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    if !cfg!(target_os = "linux") {
        return Err("this parity executable requires CUDA on Linux".into());
    }
    let args: Vec<String> = std::env::args().collect();
    let export = args
        .get(1)
        .ok_or("mpd_artifact_gpu_check_2951 EXPORT [SEQUENCES] [CONTEXT] [ABS_TOLERANCE]")?;
    let count = |index: usize, default: usize| -> Result<usize, String> {
        args.get(index).map_or(Ok(default), |v| {
            v.parse().map_err(|e| format!("argument {index}: {e}"))
        })
    };
    let sequences = count(2, 1)?;
    let context = count(3, 12)?;
    let tolerance: f64 = args.get(4).map_or(Ok(1e-8), |v| {
        v.parse().map_err(|e| format!("tolerance: {e}"))
    })?;
    if sequences == 0 || context == 0 || !tolerance.is_finite() || tolerance < 0.0 {
        return Err("positive sequences/context and finite nonnegative tolerance required".into());
    }
    let device = Device::accelerator(gam_gpu::GpuPolicy::Required)
        .map_err(|e| e.to_string())?
        .ok_or("CUDA accelerator required")?;
    if device.is_host() || !device.float64() {
        return Err("a float64 CUDA accelerator is required".into());
    }
    let imported = import_language_model(Path::new(export), sequences, context)?;
    let family = imported.family;
    let model = imported.program;
    let mut artifact = Artifact::native(&model)?;
    let (norm, input, epsilon) = model
        .nodes
        .iter()
        .enumerate()
        .find_map(|(index, node)| match node {
            Node::RmsNorm { input, epsilon } => Some((index, *input, *epsilon)),
            _ => None,
        })
        .ok_or("the exported model has no normalization node")?;
    let rule = artifact.program.rules.len();
    artifact.program.rules.push(Rule {
        name: "normalization call".into(),
        inputs: vec![model.node_interface(input).map_err(|e| e.to_string())?],
        nodes: vec![
            Node::Param { index: 0 },
            Node::RmsNorm { input: 0, epsilon },
        ],
        output: 1,
    });
    artifact.program.nodes[norm] = Node::Call {
        rule,
        arguments: vec![input],
    };
    artifact = artifact.bind("normalization call", &[input], norm)?;
    artifact.exceptions.push(Exception {
        context: contexts(&family)[0].clone(),
        node: norm,
        column: 0,
        value: 0.25,
    });
    let artifact = EncodedArtifact::of(&artifact.f32_literals()?)?.decode()?;
    let expected = artifact.execute_edited(&family, |node, value, _| {
        if node == norm || node == artifact.program.output {
            value.mapv_inplace(|v| 2.0 * v);
        }
        Ok(())
    })?;
    let resident = Resident::new(&device, &artifact)?;
    let actual = resident.forward_edited(&family, |node, trace| {
        if node == norm || node == artifact.program.output {
            let source = resident.root_value(trace, node)?;
            let mut out = device.copy(source).map_err(|e| e.to_string())?;
            device
                .axpy(&mut out, 1.0, source)
                .map_err(|e| e.to_string())?;
            Ok(Some(out))
        } else {
            Ok(None)
        }
    })?;
    let actual = device
        .download(&resident.output(&actual)?)
        .map_err(|e| e.to_string())?;
    let expected = &expected.values[artifact.program.output];
    if actual.dim() != expected.dim()
        || actual.iter().chain(expected.iter()).any(|v| !v.is_finite())
    {
        return Err("parity output dimensions differ or contain nonfinite values".into());
    }
    let gap = (&actual - expected)
        .iter()
        .map(|x| x.abs())
        .fold(0.0f64, f64::max);
    println!(
        "{}",
        json!({"device": device.name(), "backend": "CUDA required", "export": export,
        "sequences": sequences, "context": context, "rows": family.rows, "maximum_absolute_gap": gap,
        "absolute_tolerance": tolerance, "passed": gap <= tolerance, "rule_calls": 1, "exceptions": artifact.exceptions.len(),
        "terminal_readout": matches!(artifact.program.nodes[artifact.program.output], Node::Readout { .. }),
        "edited_root_nodes": [norm, artifact.program.output],
        "estimated_resident_bytes_incomplete": resident.estimated_resident_bytes(family.rows)?})
    );
    if gap > tolerance {
        return Err(format!(
            "CUDA artifact parity gap {gap} exceeds {tolerance}"
        ));
    }
    Ok(())
}
