//! Proposal-only max-row fitting of a supplied explicit nonlinear Rule/Call artifact.
//! SOURCE.bin EXTRACT.json LAYER SETTINGS.json TRAINABLE.json OUT_DIR host|cuda
//! The archive contains disjoint native train/eval arrays; no fitting oracles from a checkpoint.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    acceptance::{CostCache, structural_cost},
    artifact::Artifact,
    operator_program::{Declarations, Slot},
    resident_rule_fit::{self, Settings},
};
use ndarray::Array2;
use serde_json::{Value, json};
use std::{path::Path, process::Command, time::Instant};
fn hash(path: &Path) -> Result<String, String> {
    for (program, args) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        if let Ok(output) = Command::new(program).args(args).arg(path).output() {
            if output.status.success() {
                let text = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
                let value = text.split_whitespace().next().ok_or("empty SHA output")?;
                if value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Ok(value.to_ascii_lowercase());
                }
            }
        }
    }
    Err("SHA256 utility required".into())
}
fn integer(v: &Value, key: &str) -> Result<usize, String> {
    usize::try_from(
        v[key]
            .as_u64()
            .ok_or_else(|| format!("missing integer {key}"))?,
    )
    .map_err(|e| e.to_string())
}
fn load(
    root: &Path,
    panel: &Value,
    layer: usize,
    role: &str,
) -> Result<(Array2<f64>, Value), String> {
    let values = panel["values"].as_array().ok_or("missing panel values")?;
    let matches: Vec<_> = values
        .iter()
        .filter(|v| v["layer"].as_u64() == Some(layer as u64) && v["role"].as_str() == Some(role))
        .collect();
    if matches.len() != 1 {
        return Err(format!("expected exactly one {role} for layer {layer}"));
    }
    let descriptor = matches[0];
    let file = descriptor["file"].as_str().ok_or("missing array file")?;
    if Path::new(file).components().count() != 1
        || !matches!(
            Path::new(file).components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err("array filename must be a sibling".into());
    }
    let path = root.join(file);
    let sha = hash(&path)?;
    if descriptor["sha256"].as_str() != Some(&sha) {
        return Err(format!("SHA mismatch {}", path.display()));
    }
    let rows = integer(panel, "rows")?;
    let width = integer(descriptor, "width")?;
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let count = rows.checked_mul(width).ok_or("array shape overflow")?;
    if bytes.len() != count.checked_mul(8).ok_or("array bytes overflow")? {
        return Err("array length mismatch".into());
    }
    let data: Vec<_> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("eight bytes")))
        .collect();
    if !data.iter().all(|v| v.is_finite()) {
        return Err("nonfinite archive".into());
    }
    Ok((
        Array2::from_shape_vec((rows, width), data).map_err(|e| e.to_string())?,
        json!({"descriptor":descriptor,"path":path,"sha256":sha}),
    ))
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 8 {
        return Err("usage: mpd_resident_rule_fit_2951 SOURCE.bin EXTRACT.json LAYER SETTINGS.json TRAINABLE.json OUT_DIR host|cuda".into());
    }
    let started = Instant::now();
    let manifest_path = Path::new(&args[2]);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(manifest_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let layer: usize = args[3].parse().map_err(|e| format!("layer: {e}"))?;
    let panels = manifest["panels"].as_array().ok_or("missing panels")?;
    let named = |name: &str| -> Result<&Value, String> {
        let p: Vec<_> = panels
            .iter()
            .filter(|p| p["name"].as_str() == Some(name))
            .collect();
        if p.len() != 1 {
            return Err(format!("expected one {name} panel"));
        }
        Ok(p[0])
    };
    let train = named("train")?;
    let valid = named("eval")?;
    let root = manifest_path.parent().ok_or("manifest parent")?;
    let (x, xd) = load(root, train, layer, "input")?;
    let (y, yd) = load(root, train, layer, "write")?;
    let (vx, vxd) = load(root, valid, layer, "input")?;
    let (vy, vyd) = load(root, valid, layer, "write")?;
    if xd["sha256"] == vxd["sha256"] || yd["sha256"] == vyd["sha256"] {
        return Err("identical train/eval arrays are not an independent validation panel".into());
    }
    let settings: Settings =
        serde_json::from_slice(&std::fs::read(&args[4]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let trainable: Vec<usize> =
        serde_json::from_slice(&std::fs::read(&args[5]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let declarations = Declarations {
        domains: vec![],
        slots: vec![Slot::Raw { width: x.ncols() }],
        parameters: 0,
    };
    let source = Artifact::from_bytes(
        &std::fs::read(&args[1]).map_err(|e| e.to_string())?,
        &declarations,
    )?;
    if !source.derived.is_empty() || !source.exceptions.is_empty() {
        return Err("fitter requires explicit operators and no executable exceptions; derived constraints/exceptions cannot be silently omitted".into());
    }
    let device = match args[7].as_str() {
        "host" => Device::host(),
        "cuda" => Device::accelerator(GpuPolicy::Required)
            .map_err(|e| e.to_string())?
            .ok_or("CUDA f64 required")?,
        _ => return Err("backend must be host or cuda".into()),
    };
    let result = resident_rule_fit::fit(
        &device,
        &source.program,
        &x,
        &y,
        &vx,
        &vy,
        &trainable,
        settings.clone(),
    )?;
    let output = Path::new(&args[6]);
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let mut best_parameters = Vec::new();
    for index in &trainable {
        let gam_mpd::operator_program::OperatorBody::Dense { values, .. } =
            &result.program.operators[*index].body
        else {
            return Err("fitted parameter changed kind".into());
        };
        let path = output.join(format!("best.{index}.f64"));
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
        best_parameters.push(json!({"operator":index,"path":path,"shape":[values.nrows(),values.ncols()],"sha256":hash(&path)?,"bytes":bytes.len(),"encoding":"little-endian f64 row-major best training snapshot; source artifact supplies all unchanged structure/parameters"}));
    }
    let mut fitted = source.clone();
    fitted.program = result.program;
    let fitted = fitted.f32_literals()?;
    let bytes = fitted.to_bytes()?;
    let output = Path::new(&args[6]);
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let saved = output.join("fitted.bin");
    std::fs::write(&saved, &bytes).map_err(|e| e.to_string())?;
    // Decode the actual saved file independently, using the ordinary codec, then remeasure.
    let replay = Artifact::from_bytes(
        &std::fs::read(&saved).map_err(|e| e.to_string())?,
        &declarations,
    )?;
    let decoded_training_max = resident_rule_fit::measure(
        &device,
        &replay.program,
        &x,
        &y,
        settings.numeric_bytes,
        settings.forward_rows,
    )?;
    let decoded_validation_max = resident_rule_fit::measure(
        &device,
        &replay.program,
        &vx,
        &vy,
        settings.numeric_bytes,
        settings.forward_rows,
    )?;
    let cost = structural_cost(&replay, &mut CostCache::default())?;
    let report = json!({"scope":"Proposal fit of a supplied explicit shared nonlinear program, not automatic operation discovery, formal Local/Run acceptance, optimizer optimum or neural arithmetic certificate.","source":{"path":args[1],"sha256":hash(Path::new(&args[1]))?},"manifest":{"path":manifest_path,"sha256":hash(manifest_path)?},"layer":layer,"training_record":train["record"],"validation_record":valid["record"],"arrays":{"training_inputs":xd,"training_outputs":yd,"validation_inputs":vxd,"validation_outputs":vyd},"backend":args[7],"optimizer":result.report,"best_f64_parameter_snapshots":best_parameters,"f32_saved_replay":{"path":saved,"sha256":hash(&saved)?,"bytes":bytes.len(),"training_max_normalized_row_error":decoded_training_max,"validation_max_normalized_row_error":decoded_validation_max,"complete_c32_bits":cost.total()},"seconds":started.elapsed().as_secs_f64(),"validation_scope":"Separate archived panels with recorded provenance and checked file hashes. Different hashes alone do not establish source-token disjointness; inspect the recorded extraction protocol."});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{}", output.join("report.json").display());
    Ok(())
}
fn main() -> Result<(), String> {
    run()
}
